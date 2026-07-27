"""Prompt templates for each orchestration role.

The quality of the orchestration lives almost entirely in these prompts. They
are written to push each role to be rigorous and to emit *strict JSON* so the
control flow can act on the decisions deterministically.

Each builder returns a list[LLMMessage] ready for ``LLMClient.complete*``.
"""

from __future__ import annotations

import re
from typing import Optional

from .artifact_repair import (
    DEFAULT_REPAIR_POLICY,
    ArtifactRepairPolicy,
    ArtifactRepairRequest,
)
from .decomposition import (
    DEFAULT_DECOMPOSITION_POLICY,
    DEFAULT_PACKAGE_SIZING_POLICY,
    ArtifactExecutionScope,
)
from .intent import DeliverableContract, RequestIntent
from .llm_client import LLMMessage
from .models import Plan, SubTask, Task

# --------------------------------------------------------------------------- #
# ARTIFACT CONTRACT
# --------------------------------------------------------------------------- #
# Every worker returns a STRICT JSON artifact (not free markdown). This exact
# contract is appended to the END of every worker prompt by the framework
# (``build_worker_messages``; re-appended after web-UI sanitization in
# ``webllm.client.render_messages_for_web`` so it is never stripped). A unique
# substring lets the web layer detect a worker call and re-attach the contract.
ARTIFACT_CONTRACT_MARKER = '"key_decisions"'

WORKER_ARTIFACT_CONTRACT = (
    "Reply with a SINGLE JSON object and nothing else — no text before or after, "
    "no code fences. Use exactly these keys:\n"
    "{\n"
    '  "summary": "A 1-2 sentence summary of what was done.",\n'
    '  "key_decisions": ["List of important technical or creative decisions made"],\n'
    '  "artifacts": {\n'
    '    "filename.ext": "The actual code, text, or content generated",\n'
    '    "another_file.md": "..."\n'
    "  },\n"
    '  "confidence": 0.95\n'
    "}\n"
    'Put the COMPLETE deliverable inside "artifacts" — one entry per file or piece '
    'of content, each with a sensible filename and extension. "confidence" is your '
    'honest self-rating from 0 to 1. Escape any double quote inside a value as \\" '
    "so the JSON stays valid, and do not truncate the content."
)

# Phase 4C: a worker with an ASSIGNED ARTIFACT PACKAGE returns its produced
# artifacts as an explicit, typed LIST keyed by the manifest's artifact ids —
# ids (not paths alone) are authoritative for collection. Same envelope, same
# marker, so the web layer's contract re-attachment is unchanged; only the
# shape of "artifacts" is tightened for scoped workers.
SCOPED_WORKER_ARTIFACT_CONTRACT = (
    "Reply with a SINGLE JSON object and nothing else — no text before or after, "
    "no code fences. Use exactly these keys:\n"
    "{\n"
    '  "summary": "A 1-2 sentence summary of what was done.",\n'
    '  "key_decisions": ["List of important technical or creative decisions made"],\n'
    '  "artifacts": [\n'
    '    {"artifact_id": "the id from YOUR assigned package", '
    '"path": "the exact path declared for that artifact", '
    '"content": "the COMPLETE file contents", '
    '"language": "e.g. tsx", "complete": true}\n'
    "  ],\n"
    '  "confidence": 0.95\n'
    "}\n"
    'Return EXACTLY one entry in "artifacts" for EVERY artifact your assigned '
    "package owns — no more and no fewer. Never return an artifact owned by "
    "another package, never invent an artifact id, and never change a declared "
    'path. Each "content" must be the complete file, not a fragment, a diff, or '
    'a description. Escape any double quote inside a value as \\" so the JSON '
    "stays valid, and do not truncate the content."
)


# --------------------------------------------------------------------------- #
# ARTIFACT PLAN (Phase 4A)
# --------------------------------------------------------------------------- #
# Appended only for depth-0 requests owed concrete artifacts: code/repository
# changes, finished CONTENT, or an explicitly named architecture/data artifact.
# It asks for a small, bounded declaration of the expected deliverables and
# work packages — NEVER file contents. Kept deliberately compact so the planner
# schema does not balloon and raise JSON failure rates.
PLANNER_ARTIFACT_PLAN_RULE = """

ARTIFACT PLAN (recommended for multi-file deliverables): you may also add an \
optional top-level "artifact_plan" key to your JSON, shaped like:
{
  "title": "short deliverable name",
  "artifacts": [
    {"id": "a1", "path": "relative/path.ext", "kind": "source_file", \
"required": true, "operation": "create", "description": "purpose", \
"external": false}
  ],
  "packages": [
    {"id": "p1", "title": "short label", "kind": "implementation", \
"objective": "what this bounded package produces", "owns": ["a1"], \
"depends_on": [], "subtask_id": "s1"}
  ],
  "validations": [
    {"id": "v1", "kind": "unit_tests", "targets": ["a1"], "required": true, \
"criterion": "what passing means"}
  ]
}
Artifact "kind" is one of: source_file, test_file, config_file, document, \
data_file, directory, patch, command_result, build_result, test_result, other. \
"operation" is one of: create, modify, delete, inspect, generate. Package \
"kind" is one of: implementation, modification, integration, validation, \
review, coordination. Validation "kind" is one of: syntax, type_check, build, \
unit_tests, integration_tests, lint, schema, existence, non_empty, custom.
Hard rules for this block: NEVER include file contents, code, or generated \
text in it — paths, names and short descriptions only. Use relative paths \
(no absolute paths, no ".."). Every required artifact must be owned by \
exactly ONE package via "owns" (or marked "external": true if supplied from \
outside). Package "depends_on" lists package ids and must be acyclic. \
"subtask_id" maps a package to the delegation that performs its work. If the \
task does not decompose into concrete artifacts, omit "artifact_plan" \
entirely. Limits: at most 80 artifacts, 24 packages, and 40 validations; paths \
are at most 240 characters, and declaration text is at most 300 characters.""" + (
    # Phase 4B bounded-decomposition rules. The numeric limits interpolate the
    # ONE authoritative policy so they are never declared twice.
    "\nPackages are BOUNDED execution units. Every package that owns artifacts "
    'or claims validations MUST set "subtask_id" (only a coordination package '
    "owning nothing may stay unmapped). Never assign a large manifest "
    "wholesale to one package or delegation — split it. Keep each package "
    f"within {DEFAULT_DECOMPOSITION_POLICY.max_owned_artifacts_per_package} "
    f"owned artifacts and "
    f"{DEFAULT_DECOMPOSITION_POLICY.max_input_artifacts_per_package} inputs, "
    f"with at most {DEFAULT_DECOMPOSITION_POLICY.max_validations_per_package} "
    "validations, "
    f"and map at most "
    f"{DEFAULT_DECOMPOSITION_POLICY.max_packages_per_subtask} packages to one "
    "delegation. A manifest becomes large above "
    f"{DEFAULT_DECOMPOSITION_POLICY.single_package_manifest_threshold} "
    "ownable artifacts; no package or combined delegation may own "
    f"{DEFAULT_DECOMPOSITION_POLICY.wholesale_ownership_percent}% or more of "
    'it. Mirror every package "depends_on" in the mapped delegations\' '
    '"depends_on": a validation package runs AFTER the packages producing '
    "what it checks, and an integration package runs AFTER the packages "
    "producing its inputs."
    # Phase 4F output-aware sizing (Part I). One delegation answers in ONE chat
    # reply, so packages are sized by expected OUTPUT, not file count alone.
    "\nSize packages by expected response OUTPUT (one delegation answers in "
    "ONE reply): a large implementation file gets its OWN package; at most "
    f"{DEFAULT_PACKAGE_SIZING_POLICY.max_substantial_files_per_package} "
    "substantial source files or "
    f"{DEFAULT_PACKAGE_SIZING_POLICY.max_small_files_per_package} small "
    "config files share one package, within "
    f"~{DEFAULT_PACKAGE_SIZING_POLICY.max_package_output_chars} estimated "
    "output characters. Split a large subsystem into several dependency-aware "
    "packages. If one artifact cannot fit, redesign it into smaller logical "
    "modules or fail planning; never fragment one file across replies."
)

PLANNER_ARTIFACT_PLAN_SYSTEM_EXCEPTION = (
    "\n\nFor this request, the following optional declaration is the sole "
    "authorized exception to the exact planner JSON shape above."
    + PLANNER_ARTIFACT_PLAN_RULE
)

_EXPLICIT_ARTIFACT_OUTPUT_RE = re.compile(
    r"\b(?:sadd|adr|rfc|readme|architecture\s+(?:document|diagram)|"
    r"design\s+doc(?:ument)?|technical\s+spec(?:ification)?|blueprint|"
    r"report|document|(?:excel\s+)?workbook|spreadsheet|"
    r"openapi\s+spec(?:ification)?|(?:kubernetes\s+)?(?:yaml\s+)?manifest|"
    r"sql\s+(?:file|script)|"
    r"(?:json|csv|yaml|yml|xml|toml)\s+(?:configuration|config|data|report|file)|"
    r"(?:configuration|config|data)\s+file|patch|diff)\b|"
    r"\.(?:md|txt|json|csv|ya?ml|xml|toml|sql)\b",
    re.IGNORECASE,
)

_ARTIFACT_ACTION_RE = re.compile(
    r"\b(?:create|write|compose|produce|generate|draft|prepare|deliver|provide|output|"
    r"return|emit|make|build|add|export|save|format|give|supply|send|put|"
    r"present|reply|edit|revise|improve|update|change|rewrite|document|fix|"
    r"implement|develop|scaffold|bootstrap|migrate|set\s+up)\b",
    re.IGNORECASE,
)
_NEGATED_CLAUSE_PREFIX_RE = re.compile(
    r"\b(?:don'?t|do\s+not|never|(?:there\s+is\s+)?no\s+need\s+to|"
    r"(?:i|we|you)\s+(?:don'?t|do\s+not)\s+need\s+to|"
    r"under\s+no\s+circumstances|(?:i|we|you)\s+would\s+rather\s+not|"
    r"(?:must(?:\s+not|n'?t)|should\s+not|not\s+to)|"
    r"anything\s+(?:but|except|other\s+than))\b"
    r"[^;,.!?]*$",
    re.IGNORECASE,
)
_HYPOTHETICAL_ACTION_PREFIX_RE = re.compile(
    r"\b(?:how\s+to|how\s+(?:do|can|could|would|should)\s+(?:i|we|you)|"
    r"how\s+(?:one|someone|we|you)\s+"
    r"(?:might|could|would|should)|how\s+should\s+(?:we|i|you)|"
    r"whether\s+(?:we|i|you)\s+should|(?:should|could|would)\s+(?:we|i)|"
    r"(?:can|may)\s+(?:i|we)|(?:can|could|would|should|may)\s+(?:they|he|she|it)|"
    r"should\s+you|(?:i|we|you|they|he|she|it)\s+(?:might|may)|"
    r"does\s+(?:this|that|the)\s+(?:tool|system|app|application|service|code|"
    r"function|script|program|command|api)|"
    r"do\s+(?:i|we|you)\s+need\s+to|would\s+it|would\s+it\s+make\s+sense\s+to|"
    r"what\s+if\s+(?:i|we|you)|(?:consider(?:ing)?|decid(?:e|ing)|"
    r"(?:tell|advise)\s+me)\s+whether\s+to|"
    r"(?:explain|discuss)\s+why\s+(?:i|we|you)\s+should|what\s+does|"
    r"(?:explain|define|interpret)\s+(?:the\s+)?(?:phrase|instruction|request)|"
    r"review\s+(?:the\s+)?(?:requirement|instruction|request)\s+to|"
    r"(?:review|assess|evaluate|discuss)\s+whether\s+to|"
    r"(?:tell|advise)\s+me\s+whether\s+[^;,.!?]{0,40}\bshould|"
    r"(?:provide|give)\s+(?:me\s+)?(?:instructions|guidance|steps|examples)\s+"
    r"(?:of\s+how\s+to|(?:on\s+)?how\s+to|to))\b"
    r"[^;,.!?]*$",
    re.IGNORECASE,
)
_EXTENDED_NEGATED_ACTION_PREFIX_RE = re.compile(
    r"(?:\b(?:don'?t|do\s+not|never)\s*,[^;.!?]{0,70},?\s*|"
    r"\bthere\s+is\s+no\s+need\s+to\s*)$",
    re.IGNORECASE,
)
_EXCLUDED_ARTIFACT_PREFIX_RE = re.compile(
    r"\b(?:no|not(?!\s+only\b)|without|anything\s+(?:but|except|other\s+than)|"
    r"instead\s+of|rather\s+than|neither|except|excluding|exclude|avoid|"
    r"prohibited|disallowed)\b[^;,.!?]*$",
    re.IGNORECASE,
)
_EXCLUDED_ARTIFACT_SUFFIX_RE = re.compile(
    r"^(?:\.[a-z0-9]+)?\s*(?:"
    r"(?:(?:is|are|was|were|should\s+be)\s+)?"
    r"(?:excluded|forbidden|unwanted|omitted|avoided|prohibited|disallowed|"
    r"not\s+(?:allowed|wanted|required|requested|included|desired|produced))|"
    r"(?:must|should)\s+not\s+be\s+"
    r"(?:produced|provided|returned|created|included|allowed))\b",
    re.IGNORECASE,
)


def _clause_prefix(text: str, position: int) -> str:
    start = max(text.rfind(mark, 0, position) for mark in ";,.!?") + 1
    return text[start:position]


def _artifact_reference_is_excluded(text: str, match: re.Match[str]) -> bool:
    prefix = _clause_prefix(text, match.start())
    suffix = text[match.end(): match.end() + 80]
    if _NEGATED_CLAUSE_PREFIX_RE.search(prefix):
        return True

    # In ``not a README but a research report`` the contrast begins a positive
    # alternative. Do not let the first reference's exclusion swallow it.
    contrasts = list(re.finditer(r"\bbut\b", prefix, re.IGNORECASE))
    if contrasts:
        contrast = contrasts[-1]
        before = prefix[:contrast.start()]
        if (
            re.search(r"\b(?:no|not|without|neither)\b", before, re.IGNORECASE)
            or _EXPLICIT_ARTIFACT_OUTPUT_RE.search(before)
        ):
            prefix = prefix[contrast.end():]
    if _EXCLUDED_ARTIFACT_PREFIX_RE.search(prefix):
        return True
    if _EXCLUDED_ARTIFACT_SUFFIX_RE.search(suffix):
        return True

    # A filename dot splits the ordinary prefix in expressions such as
    # ``Neither README.md nor report.csv``. Track the wider neither/nor scope,
    # while allowing a later ``but ...`` positive alternative to reset it.
    broad_prefix = text[:match.start()]
    last_neither = max(
        (candidate.start() for candidate in re.finditer(
            r"\bneither\b", broad_prefix, re.IGNORECASE
        )),
        default=-1,
    )
    last_reset = max(
        (candidate.start() for candidate in re.finditer(
            r"\bbut\b|[;!?]|\.(?=\s+[A-Z])", broad_prefix, re.IGNORECASE
        )),
        default=-1,
    )
    return last_neither > last_reset


def _has_nonnegated_artifact_name(text: str) -> bool:
    for match in _EXPLICIT_ARTIFACT_OUTPUT_RE.finditer(text):
        if not _artifact_reference_is_excluded(text, match):
            return True
    return False


def _action_is_non_output(text: str, action: re.Match[str]) -> bool:
    prefix = _clause_prefix(text, action.start())
    extended_prefix = text[max(0, action.start() - 120):action.start()]
    if _NEGATED_CLAUSE_PREFIX_RE.search(prefix):
        return True
    if _EXTENDED_NEGATED_ACTION_PREFIX_RE.search(extended_prefix):
        return True
    if _HYPOTHETICAL_ACTION_PREFIX_RE.search(prefix):
        return True
    if action.group(0).lower() in {"provide", "give"} and re.match(
        r"\s+(?:me\s+)?(?:instructions|guidance|steps|examples)\s+"
        r"(?:of\s+how\s+to|(?:on\s+)?how\s+to|to)\b",
        text[action.end():],
        re.IGNORECASE,
    ):
        return True
    # ``create no README`` places the prohibition after the action verb.
    return bool(re.match(r"\s+no\b", text[action.end():], re.IGNORECASE))


def _has_positive_content_action(text: str) -> bool:
    """Require an actual output action for a Phase 3 CONTENT classification."""
    for action in _ARTIFACT_ACTION_RE.finditer(text):
        # Do not read the stem of a filename such as ``output.csv`` as a verb.
        if action.end() < len(text) and text[action.end()] == ".":
            continue
        if not _action_is_non_output(text, action):
            return True
    return False


def _has_action_verb(text: str) -> bool:
    """Whether this surface contains an action token rather than a filename stem."""
    return any(
        action.end() >= len(text) or text[action.end()] != "."
        for action in _ARTIFACT_ACTION_RE.finditer(text)
    )


def _has_positive_artifact_action(text: str) -> bool:
    """Match an explicit output action without treating prohibitions as work."""
    for action in _ARTIFACT_ACTION_RE.finditer(text):
        # Do not read the stem of a filename such as ``output.csv`` as a verb.
        if action.end() < len(text) and text[action.end()] == ".":
            continue
        if _action_is_non_output(text, action):
            continue
        for artifact in _EXPLICIT_ARTIFACT_OUTPUT_RE.finditer(
            text, action.end(), min(len(text), action.end() + 100)
        ):
            between = text[action.end():artifact.start()]
            # Commas may delimit parenthetical wording within one instruction.
            if re.search(r"[;!?]|\.(?=\s|$)", between):
                break
            if not _artifact_reference_is_excluded(text, artifact):
                return True
    return False


def artifact_planning_enabled(task: Task, depth: int) -> bool:
    """Whether this root request is owed a concrete artifact declaration."""
    contract = task.contract
    if depth != 0 or contract is None:
        return False
    # Phase 4B: a task confined to an inherited artifact scope (a recursive
    # child of a scoped subtask) must never declare a NEW root manifest or
    # redefine ownership — its permitted outputs are already fixed.
    if getattr(task, "execution_scope", None) is not None:
        return False
    action_surfaces = tuple(
        str(part)
        for part in (task.prompt, *task.constraints)
        if str(part).strip()
    )
    # ``desired_output`` is already an authoritative output slot, so an
    # explicitly named structured artifact there governs regardless of whether
    # the surrounding request is research, review, explanation, or design.
    if _has_nonnegated_artifact_name(str(task.desired_output or "")):
        return True
    if contract.code_required or contract.repo_changes_required:
        if any(_has_positive_content_action(surface)
               for surface in action_surfaces):
            return True
        # Preserve symptom-only DEBUG activation, but do not let Phase 3's
        # action flag override a locally explicit permission question,
        # hypothetical, description, or prohibition.
        return not any(_has_action_verb(surface) for surface in action_surfaces)
    if contract.intent is RequestIntent.CONTENT:
        # Phase 3 identifies the deliverable category; this local action check
        # distinguishes a finished written artifact from guidance about making one.
        content_surfaces = action_surfaces + (
            (str(task.desired_output),)
            if str(task.desired_output or "").strip()
            else ()
        )
        return any(_has_positive_content_action(surface)
                   for surface in content_surfaces)
    return any(_has_positive_artifact_action(surface)
               for surface in action_surfaces)


# --------------------------------------------------------------------------- #
# PLANNER
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = """You are the MANAGER of a multi-agent orchestration system. \
You do NOT answer the task yourself. Your job is to (a) split the task into a \
minimal dependency graph of subtasks, (b) INTELLIGENTLY ROUTE each subtask to the \
single best-suited model from the provided pool, and (c) note the order in which \
DOZEN deterministically assembles the results into one final answer.

You will be given an `AVAILABLE MODELS` list. Each line is:
  - <model_name>: <capability profile> (tier N/5)
`<model_name>` is the exact identifier you MUST copy into `assigned_model`.

INTELLIGENT ROUTING — THE CORE OF YOUR JOB:
1. This is NOT round-robin. Do NOT default every subtask to the same model. \
Different subtasks should usually go to different models when their strengths \
differ.
2. For EACH subtask, read its objective, then scan EVERY model's profile and pick \
the ONE whose stated strengths best match that specific objective. Example: route \
coding/logic/architecture work to a model described as strong at coding/reasoning; \
route research/creative/summarization work to a model described as strong at \
those.
3. Break ties by tier (prefer the higher tier for harder/`difficulty`>=4 work) \
and by keeping a cheaper/faster model for easy, low-stakes subtasks.
4. You may ONLY use model names that appear verbatim in `AVAILABLE MODELS`. Never \
invent a name. If only one model exists, assign it to every subtask.
5. For every subtask you MUST justify the choice in `model_selection_reasoning`: \
name the subtask's key demand and the specific strength in THAT model's profile \
that makes it the best fit (and, if relevant, why the others are worse). Do not \
write generic filler like "it is good"; reference the profile.

DECOMPOSITION RULES:
6. Separate by CONCERN — each subtask has ONE clear objective and a self-contained \
output. Use the FEWEST subtasks that fully cover the task.
7. If the task is trivial, set `direct_answer` to the full answer and leave \
`delegations` empty. Otherwise set `direct_answer` to null and provide a \
non-empty `delegations` list. Never both.
8. `depends_on` lists the ids of subtasks whose output THIS one needs; \
independent subtasks run in PARALLEL. The graph must be acyclic.
9. Write each `instruction` EXACTLY as a normal person would type it into a chat \
box: a direct, natural request. It must be self-contained — restate any needed \
context in plain language — and it is sent to a consumer chat UI whose safety \
filters REJECT prompts that look like system-prompt injection. Therefore the \
`instruction` MUST NOT contain ANY of: persona/roleplay commands ("You are an \
AI…", "Act as…", "As an AI…"); references to subtasks, sub-agents, "the model", \
orchestration, pipelines, or being part of a system; meta-labels ("Your subtask \
is:", "Role:", "Target model:", "Instruction:"); or any JSON, markdown, code \
fences, or backticks. Just ask for the thing, like a human.
   Also give each `instruction` STRICT SCOPE BOUNDARIES so subtasks never overlap \
or duplicate each other's work. State plainly what this request should focus on \
and, where useful, what it must NOT do — phrased as a natural user constraint, \
NEVER as a reference to other subtasks, agents, or the pipeline. Match the \
exclusion to the assigned model's job: tell a writing task not to write code, \
tell a coding task not to add prose, tell a research task not to draft the final \
copy, etc.
   Bad:  "You are an AI copywriter. Your subtask is to write a blog post about fitness."
   Good (writing): "Please write a highly engaging marketing blog post about a new AI-powered fitness startup. Do NOT include any code or technical setup steps — focus only on the writing."
   Good (coding):  "Please write a complete Python script that scrapes the page titles from a list of URLs. Only provide the code with brief inline comments — do not add marketing copy or a long written explanation."
10. `synthesis_strategy` is a SHORT ordering note for DOZEN's own DETERMINISTIC \
assembler — not an instruction to any model. State only the natural order the \
finished parts should appear in and which subtask is authoritative for which \
part. DOZEN combines the validated results itself, in code, with NO model merge \
step: never describe a model that repairs, completes, merges, reformats, or \
reviews-and-combines the other subtasks' outputs, and never create a subtask to \
do so (rule 13). Put each part's own final output format inside that subtask's \
`expected_output`, not here.

ARTIFACT-BASED WORKER OUTPUTS:
11. Every worker returns a STRICT JSON ARTIFACT, not free text. The system \
automatically appends this exact output contract to the END of each worker's \
prompt, so you must NOT paste it into `instruction` yourself — keep `instruction` \
clean and human (rule 9). The contract each worker receives is:
   { "summary": "1-2 sentence summary", "key_decisions": ["..."], "artifacts": \
{ "filename.ext": "the actual generated content" }, "confidence": 0.0-1.0 }
12. Design AROUND the artifact shape: set each `expected_output` to describe what \
should land inside `artifacts` (e.g. "a complete app.py in artifacts"). DOZEN \
assembles the final deliverable from those `artifacts` values deterministically, \
in manifest order — you do NOT delegate that assembly to any model.

NO SYNTHESIS / MERGE DELEGATIONS:
13. NEVER create a subtask/delegation whose job is to merge, integrate, \
assemble, consolidate, synthesize, stitch, or "review and combine" the OTHER \
subtasks' outputs — and never a "final integration", "final assembly", "final \
integration review", "final synthesis", or "return part N of M" step. DOZEN \
assembles every validated result deterministically in code; a delegation that \
hands one model the other subtasks' outputs to merge is INVALID and will be \
rejected. Every delegation must produce its OWN self-contained part of the \
deliverable, never the combined whole.

OUTPUT CONTRACT:
- Respond with a SINGLE valid JSON object and NOTHING else: no prose, no markdown, \
no code fences, no comments (JSON allows neither `//` nor `/* */`).
- Use double quotes for all keys/strings; lowercase `null`/`true`/`false`; no \
trailing commas.
- CRITICAL — ESCAPE QUOTES INSIDE STRING VALUES. A string value is delimited by \
double quotes, so any double quote that appears INSIDE the text (e.g. in \
`direct_answer`, `instruction`, or `analysis`) WILL break the JSON unless it is \
escaped. You MUST do one of these for every internal quotation:
    1. Escape each internal double quote as \\" — e.g. \
"direct_answer": "The log said \\"FATAL ERROR\\" today.", or
    2. Use single quotes ' for the inner quotation instead — e.g. \
"direct_answer": "The log said 'FATAL ERROR' today."
  Also escape other JSON control characters inside strings: use \\\\ for a \
backslash and \\n for a newline. Never emit a raw, unescaped " in the middle of a \
string value.
- Keep `synthesis_strategy` to a brief, plain ordering note; it authorizes no \
model to merge, rewrite, or repair any output (DOZEN assembles the validated \
results deterministically, in code).

Return EXACTLY this shape (types shown, not literals). Subtask ids are "s1", \
"s2", ... and are reused in `depends_on`:
{
  "analysis": "one short sentence on your delegation + routing approach",
  "direct_answer": "the full answer, or null if you are decomposing",
  "delegations": [
    {
      "id": "s1",
      "title": "short label",
      "instruction": "self-contained instruction for the assigned model",
      "assigned_model": "<exact model_name from AVAILABLE MODELS>",
      "model_selection_reasoning": "why this model's profile best fits this subtask",
      "required_capabilities": ["reasoning"],
      "depends_on": [],
      "success_criteria": "what makes the output acceptable",
      "expected_output": "what should land inside the worker's artifacts (e.g. 'a complete app.py')",
      "complex": false,
      "difficulty": 3
    }
  ],
  "synthesis_strategy": "a brief note on the natural order of the finished parts for DOZEN's deterministic assembler (no model merge step)"
}"""


def build_planner_messages(
    task: Task,
    depth: int,
    max_depth: int,
    available_models: str = "",
    repair_feedback: str = "",
) -> list[LLMMessage]:
    artifact_enabled = artifact_planning_enabled(task, depth)
    depth_note = ""
    if depth >= max_depth:
        depth_note = (
            "\n\nIMPORTANT: Recursion depth limit reached. Do NOT mark any subtask "
            "as complex; every subtask must be directly solvable by a single model."
        )
    models_block = available_models.strip() or (
        "(No model catalog supplied — assign the single available model.)"
    )
    # Phase 3: the deliverable contract is the authoritative statement of what
    # this request owes the user. For code-bearing contracts it makes a plan
    # made only of "describe / recommend / explain" work explicitly invalid.
    contract_block = ""
    if task.contract is not None:
        contract_block = "\n\n" + task.contract.to_brief()
        if not task.contract.architecture_only_allowed:
            contract_block += (
                "\n\nPLAN VALIDITY RULE: a plan whose subtasks only describe the "
                "architecture, recommend libraries, explain implementation steps, "
                "write a design document, discuss possible code, or claim the work "
                "is too large is INVALID and will be rejected. At least one subtask "
                "MUST produce the actual artifact (write the source files / apply "
                "the change / deliver the fix)"
                + (", and a subtask MUST cover tests or validation."
                   if task.contract.tests_required else ".")
            )
        # Phase 4F (Part G): a code-only request cannot be planned without the
        # typed artifact declaration — without it there is nothing to assemble.
        if getattr(task.contract, "code_only", False) and artifact_enabled:
            contract_block += (
                "\n\nCODE-ONLY DELIVERY RULE: the user asked for code only. "
                'You MUST include the "artifact_plan" block declaring EVERY '
                "required file and the bounded work package that owns it. Do "
                "not plan subtasks whose output is introductory or concluding "
                "prose — the final deliverable is the source files themselves."
            )
        # Phase 4A's optional declaration is added once at system priority
        # below. Recursive children and non-artifact requests retain the exact
        # legacy prompt.
    # Phase 4B: a recursive child of a scoped subtask plans WITHIN its
    # inherited artifact scope — only its assigned packages, never the whole
    # root manifest (and never a new artifact schema; see the gate above).
    scope = getattr(task, "execution_scope", None)
    scope_block = ""
    if scope is not None:
        scope_block = "\n\n" + scope.to_planner_block(
            output_required=task.execution_scope_output_required
        )
    repair_block = ""
    if repair_feedback.strip():
        repair_block = (
            "\n\nYOUR PREVIOUS PLAN WAS REJECTED. Fix these problems and re-plan:\n"
            f"{repair_feedback.strip()}"
        )
    user = (
        "AVAILABLE MODELS (assign each subtask to the best-fit name below):\n"
        f"{models_block}\n\n"
        "Decompose, intelligently route, and plan the following task.\n\n"
        f"{task.to_brief()}"
        f"{contract_block}"
        f"{scope_block}"
        f"\n\n(Current orchestration depth: {depth}/{max_depth}.)"
        + depth_note + repair_block
    )
    return [
        LLMMessage(
            "system",
            PLANNER_SYSTEM
            + (PLANNER_ARTIFACT_PLAN_SYSTEM_EXCEPTION if artifact_enabled else ""),
        ),
        LLMMessage("user", user),
    ]


# --------------------------------------------------------------------------- #
# ROUTER
# --------------------------------------------------------------------------- #
ROUTER_SYSTEM = """You are the Router of a multi-agent orchestration system. \
Given one subtask and a catalog of available agents (models) with their \
strengths, tier, cost and latency, choose the SINGLE best agent to execute it.

Decision rules:
1. Match the subtask's required capabilities to agent strengths first.
2. For high-difficulty subtasks, prefer higher-tier agents even at higher cost.
3. For easy/low-stakes subtasks, prefer cheaper/faster agents.
4. Respect context-size needs (long inputs need long-context agents).
5. Break ties by lower cost, then lower latency.

Respond with STRICT JSON ONLY:
{ "agent": "<exact agent name>", "reason": str }"""


def build_router_messages(
    subtask: SubTask, agent_catalog: str, context_chars: int
) -> list[LLMMessage]:
    caps = ", ".join(subtask.required_capabilities) or "general"
    user = (
        f"SUBTASK: {subtask.title}\n"
        f"Instruction: {subtask.instruction}\n"
        f"Required capabilities: {caps}\n"
        f"Difficulty (1-5): {subtask.difficulty}\n"
        f"Approx input size (chars): {context_chars}\n\n"
        f"AVAILABLE AGENTS:\n{agent_catalog}\n\n"
        "Pick the single best agent."
    )
    return [
        LLMMessage("system", ROUTER_SYSTEM),
        LLMMessage("user", user),
    ]


# --------------------------------------------------------------------------- #
# WORKER
# --------------------------------------------------------------------------- #
WORKER_SYSTEM = """You are an expert Worker agent in a multi-agent system. You \
are given ONE subtask plus the outputs of its prerequisite subtasks. Do the \
subtask thoroughly and correctly. Produce exactly the requested output and \
nothing extraneous. If prerequisite outputs are provided, use them faithfully \
and do not contradict them without clear justification."""


def build_worker_messages(
    task: Task,
    subtask: SubTask,
    dependency_outputs: dict[str, str],
    repair_feedback: str = "",
    scope: Optional[ArtifactExecutionScope] = None,
    repair: Optional[ArtifactRepairRequest] = None,
    repair_policy: Optional[ArtifactRepairPolicy] = None,
) -> list[LLMMessage]:
    parts = [
        f"OVERALL TASK (for context):\n{task.prompt.strip()}",
        f"\nYOUR SUBTASK: {subtask.title}\n{subtask.instruction.strip()}",
    ]
    # Phase 3: every worker receives the authoritative output mode, including
    # prose/no-change modes. Legacy callers with no contract remain byte-identical.
    if task.contract is not None:
        requirement = f"\nDELIVERABLE REQUIREMENT: {task.contract.short_line()}"
        if task.contract.code_required or task.contract.repo_changes_required:
            requirement += (
                " If your subtask is the part that produces code or changes, "
                "output the complete artifact itself — descriptions of what the "
                "code would look like, or a plan to write it, are not acceptable "
                "substitutes."
            )
        parts.insert(1, requirement)
    if task.constraints:
        shown = [" ".join(str(item).split())[:120] for item in task.constraints[:4]]
        if len(task.constraints) > 4:
            shown.append(f"… (+{len(task.constraints) - 4} more constraints)")
        parts.insert(1, "\nTASK CONSTRAINTS:\n" + "\n".join(f"- {x}" for x in shown))
    # Additive (Phase 1.4): background context — e.g. the conversation so far —
    # now reaches workers too, not just the planner/synthesizer briefs. Empty
    # context (the pre-1.4 web default) produces the exact pre-1.4 prompt.
    if task.context.strip():
        parts.insert(
            1,
            "\nBACKGROUND CONTEXT (earlier conversation / supplied context — "
            f"use it when the subtask refers to it):\n{task.context.strip()}",
        )
    if subtask.expected_output.strip():
        parts.append(f"\nEXPECTED OUTPUT FORMAT:\n{subtask.expected_output.strip()}")
    if subtask.success_criteria.strip():
        parts.append(f"\nSUCCESS CRITERIA:\n{subtask.success_criteria.strip()}")
    # Phase 4B: the subtask's OWN bounded artifact scope — its assigned
    # packages, owned paths, inputs and validations, never the full manifest
    # and never any file contents. Absent scope keeps the legacy prompt
    # byte-identical.
    if scope is not None:
        parts.append("\n" + scope.to_worker_block())
    # Phase 4E: on a targeted repair attempt the ``scope`` above is the NARROW
    # repair scope (only the outstanding artifacts), and this block names the
    # preserved artifacts — by path and bounded hash, never by content — that
    # must not come back. The first attempt passes ``repair=None`` and its prompt
    # is byte-identical to Phase 4D.
    if repair is not None:
        parts.append(
            "\n" + repair.to_worker_block(repair_policy or DEFAULT_REPAIR_POLICY)
        )
    if dependency_outputs:
        dep_block = "\n\n".join(
            f"--- Output of prerequisite [{title}] ---\n{out}"
            for title, out in dependency_outputs.items()
        )
        parts.append(f"\nPREREQUISITE OUTPUTS:\n{dep_block}")
    if repair_feedback.strip():
        parts.append(
            "\nREVISION REQUIRED. A previous attempt was rejected by the verifier "
            f"for these reasons. Fix them:\n{repair_feedback.strip()}"
        )
    # Artifact-based output: append the strict JSON contract as the FINAL block so
    # the worker returns a predictable artifact object, not unstructured markdown.
    # A worker that owns manifest artifacts (Phase 4C) gets the typed, id-keyed
    # variant so its submissions can be collected; every other worker keeps the
    # original contract byte-for-byte.
    parts.append("\n" + (
        SCOPED_WORKER_ARTIFACT_CONTRACT
        if scope is not None and scope.output_artifact_ids
        else WORKER_ARTIFACT_CONTRACT
    ))
    return [
        LLMMessage("system", WORKER_SYSTEM),
        LLMMessage("user", "\n".join(parts)),
    ]


# --------------------------------------------------------------------------- #
# VERIFIER
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM = """You are the Verifier of a multi-agent system. You critically \
evaluate whether a worker's output satisfies its subtask's success criteria. Be \
strict, specific, and fair. Reward correctness and completeness; penalize \
hallucination, missing requirements, and format violations.

Respond with STRICT JSON ONLY:
{
  "passed": bool,        // true if the output is acceptable as-is
  "score": float,        // 0.0..1.0 quality score
  "feedback": str        // concrete, actionable issues to fix if not passed
}"""


def build_verifier_messages(
    subtask: SubTask,
    output: str,
    dependency_outputs: dict[str, str],
    contract: Optional[DeliverableContract] = None,
    scope: Optional[ArtifactExecutionScope] = None,
) -> list[LLMMessage]:
    dep_note = ""
    if dependency_outputs:
        dep_note = "\n\nPrerequisite outputs the answer was allowed to rely on:\n" + (
            "\n".join(f"[{t}]: {o[:500]}" for t, o in dependency_outputs.items())
        )
    # Phase 3: judge against the request's deliverable mode. Note the symmetry —
    # architecture prose must not pass a code-required task, and a review must
    # NOT be failed merely for containing no code.
    contract_note = ""
    if contract is not None:
        brief = contract.to_brief()
        if contract.code_required or contract.repo_changes_required:
            contract_note = (
                f"\n\n{brief}\nIf this "
                "subtask was supposed to produce code or a change, an answer that "
                "only describes, plans, or explains the work FAILS — score it low "
                "and say what artifact is missing. (If this particular subtask is "
                "genuinely a research/design step feeding a later coding subtask, "
                "judge it on its own terms.)"
            )
        else:
            contract_note = (
                f"\n\n{brief}\nDo NOT penalize "
                "this answer for containing no code — code is not what this request "
                "asked for."
            )
    # Phase 4B: the same bounded artifact scope the worker received, so the
    # verifier judges coverage of the ASSIGNED packages only. It evaluates the
    # response text alone — no filesystem, no assembly, no written-file claims.
    scope_note = ""
    if scope is not None:
        scope_note = "\n\n" + scope.to_verifier_block()
    user = (
        f"SUBTASK: {subtask.title}\n"
        f"Instruction: {subtask.instruction}\n"
        f"Success criteria: {subtask.success_criteria or '(use your judgment)'}\n"
        f"Expected output: {subtask.expected_output or '(unspecified)'}"
        f"{contract_note}"
        f"{scope_note}"
        f"{dep_note}\n\n"
        f"WORKER OUTPUT TO EVALUATE:\n{output}\n\n"
        "Evaluate it."
    )
    return [
        LLMMessage("system", VERIFIER_SYSTEM),
        LLMMessage("user", user),
    ]


# --------------------------------------------------------------------------- #
# SYNTHESIZER
# --------------------------------------------------------------------------- #
SYNTHESIZER_SYSTEM = """You are the SYNTHESIZER and FINAL QA REVIEWER of a \
multi-agent system. You are given the original task, the manager's synthesis \
strategy, and the outputs that worker models (e.g. one subtask handled by Gemini, \
another by GPT) returned for each subtask. Stitch them into ONE coherent, \
complete, correct, working final answer for the user.

IMPORTANT: these worker outputs were scraped from web chat UIs, so they are often \
TRUNCATED (cut off mid-sentence or mid-code) or contain minor syntax errors, bad \
indentation, or unbalanced brackets. You are FULLY AUTHORIZED AND EXPECTED to fix \
all of this. A broken artifact is never acceptable just because a sub-agent \
produced it. Never refuse to merge because an input is imperfect — repair it.

How to treat the delegated subtask outputs:
1. Trust each output's SUBSTANCE and decisions as authoritative for its scope — \
do not re-solve the subtask or change its intent or approach. But you MUST fix its \
mechanical defects (see QA duties below).
2. Follow the manager's synthesis strategy for ordering and precedence.
3. Merge faithfully: keep all required content, remove duplication, and weave the \
pieces into one seamless answer (not a list of "Subtask 1 said…, Subtask 2 said…").
4. If two outputs genuinely conflict, resolve it using the strategy's precedence \
(or the more specific/better-supported one) and state the resolution briefly.

QA / REPAIR DUTIES (this is the difference between a draft and a deliverable):
5. If any code or text is TRUNCATED or cut off, intelligently complete it and \
close every open function, loop, bracket, parenthesis, quote, tag, or data \
structure so the result is whole.
6. Fix syntax errors, broken or missing indentation, and formatting so that all \
code is valid and runnable and any structured data (JSON/CSV/tables) parses.
7. Never refuse, never apologize, and never emit placeholders like "// rest of \
code here" or "...". If something is incomplete, finish it yourself so the final \
deliverable is whole and works end to end.

FULL COVERAGE OF MULTI-PART TASKS (violating this is the worst possible failure):
8. If the original task contains multiple tasks, parts, questions, or sections, \
the final answer MUST contain EVERY part's deliverable, in the task's order, \
each under a clear heading (e.g. "## Task 3 — Frontend"). Count the parts in \
the original task and check your answer covers every one before finishing. \
Dropping a section is never acceptable, no matter how long the answer gets.
9. A formatting constraint stated for ONE part — such as "output only raw \
JSON", "no markdown", "exactly 250 words", or "no conversational text" — \
applies ONLY inside that part's own section. It NEVER governs the rest of the \
answer. In particular, do NOT let the final part's format rule (e.g. "output \
only the raw JSON") collapse the entire reply into just that part: emit all \
other sections normally, then satisfy that constraint within its own section.
10. Include code files, configs, essays and other artifacts WHOLE in their \
sections (code inside fenced blocks) — never replaced by one-line summaries or \
descriptions of what the artifact "would" contain.

11. Apart from the section headings, output ONLY the final deliverable — no \
preamble, no meta-commentary, no mention of subtasks, agents, or this process.
12. Do not invent unsupported facts — but completing obvious truncation and \
fixing syntax/formatting is REQUIRED repair work, not invention."""


def build_synthesizer_messages(
    task: Task, synthesis_strategy: str, ordered_outputs: list[tuple[str, str]]
) -> list[LLMMessage]:
    body = "\n\n".join(
        f"--- Subtask [{title}] output ---\n{out}" for title, out in ordered_outputs
    )
    # Phase 3: preserve the requested OUTPUT MODE. Without this the synthesizer
    # will happily turn concrete worker artifacts into a tidy architecture essay.
    contract_block = ""
    mode_rule = ""
    if task.contract is not None:
        contract_block = task.contract.to_brief() + "\n\n"
        if task.contract.code_required or task.contract.repo_changes_required:
            mode_rule = (
                "\n\nOUTPUT MODE (mandatory): this request's deliverable is working "
                "code/changes. Carry every concrete artifact the workers produced "
                "into the final answer IN FULL (complete files in fenced blocks). "
                "Do NOT convert them into an architecture essay, a summary of the "
                "approach, or a list of recommended libraries, and never state that "
                "code is omitted or out of scope. Keep delivered implementation, "
                "supporting explanation, validation results and genuine limitations "
                "clearly separated — with the implementation first."
            )
    scope_block = ""
    if task.execution_scope is not None:
        scope_block = (
            "\n\n"
            + task.execution_scope.to_synthesizer_block(
                output_required=task.execution_scope_output_required
            )
        )
    user = (
        f"ORIGINAL TASK:\n{task.to_brief()}\n\n"
        f"{contract_block}"
        f"SYNTHESIS STRATEGY:\n{synthesis_strategy}\n\n"
        f"SUBTASK OUTPUTS (in execution order):\n{body}\n\n"
        "Produce the complete final answer now. It must cover EVERY part of the "
        "original task and include the full content of every subtask output "
        "above (repaired where defective) — any per-part formatting rule "
        "applies only within that part's own section, not to the rest of the "
        "answer."
        f"{mode_rule}{scope_block}"
    )
    return [
        LLMMessage("system", SYNTHESIZER_SYSTEM),
        LLMMessage("user", user),
    ]
