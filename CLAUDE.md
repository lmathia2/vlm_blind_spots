# Claude Code Configuration

@import PROJECT.md

---

## Planning

- For any task involving multiple files or architectural changes: write a plan to `tasks/todo.md` BEFORE writing code.
- Plans must include: goal, approach, files to modify, risks/edge cases, and verification steps.
- If implementation diverges from the plan, STOP and update the plan before continuing.
- For single-file, well-scoped changes: skip the plan and just do it.
- If something breaks mid-implementation, stop and re-plan — don't push through a failing approach.

## Subagent Usage

- Use subagents for: file exploration, reading documentation, running test suites, researching unfamiliar APIs, generating boilerplate, and parallel analysis.
- Do NOT use subagents for: core architectural decisions or changes requiring full project context.
- Each subagent gets a single, clearly scoped task with an explicit expected output.
- Prefer subagents over polluting the main context with exploratory work.

## Task Tracking

- Write plans to `tasks/todo.md` with checkable items (`- [ ]`).
- Check in with the user before starting implementation on non-trivial plans.
- Mark items complete as you go. Add a review section when finished.
- After any correction from the user, update `tasks/lessons.md` (see format below).

## Lessons File (`tasks/lessons.md`)

Each entry follows this format:

```
### [Date] Short description
- **Pattern**: What went wrong (concrete, specific)
- **Rule**: The check or behavior that prevents recurrence
- **Example**: Before/after or a brief code snippet
```

- Review this file at the start of every session.
- Prune entries that are no longer relevant to the project.

## Verification

- Never mark a task complete without proving it works.
- Run the full test suite, not just new tests, before marking complete.
- For UI changes: visually verify the result.
- For API changes: test both happy path and at least one error case.
- For bug fixes: confirm the original reproduction case now passes.
- Provide a brief summary of what changed and why.

## Code Quality

- Make every change as simple as possible. Touch minimal code.
- Always trace bugs to root cause before patching. No band-aid fixes.
- If a function exceeds ~40 lines, consider decomposition.
- If a change introduces duplication, extract it.
- Skip optimization for throwaway scripts or one-off tasks — don't over-engineer.
- Never suppress errors with empty catch blocks.
- Never leave TODO comments without a clear action and context.

## Code Style

- Follow existing patterns in the codebase. Match naming conventions, import style, and file organization already in use.
- When no precedent exists, prefer explicit over clever.
- Write comments for *why*, not *what*. The code should explain itself.
- Keep functions focused on a single responsibility.

## Communication

- When uncertain between approaches, present both with tradeoffs — don't pick one silently.
- If a task is blocked or requirements are ambiguous, say so immediately rather than guessing.
- Keep explanations concise. Prefer code comments over long chat messages for implementation details.
- Give a high-level summary at each step, not a line-by-line walkthrough.

## Git Conventions

- Commit messages: imperative mood, under 72 characters, reference issue number if applicable.
- One logical change per commit.
- Never commit secrets, API keys, `.env` files, or credentials.

## Boundaries

- Do not modify files outside the scope of the current task without explicit approval.
- Do not refactor unrelated code while fixing a bug.
- Do not add dependencies without mentioning it to the user.
- If you're unsure whether something is in scope, ask.

## Autonomous Bug Fixing

- When given a bug report: reproduce it, find the root cause, fix it, and verify — without asking for hand-holding.
- Point at logs, errors, and failing tests as evidence.
- If the bug cannot be reproduced or the cause is unclear, report findings instead of guessing at a fix.