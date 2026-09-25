# Security policy

## Reporting a vulnerability

Email **security@aethics.com**, or use
**[GitHub's private vulnerability reporting](https://github.com/Aethics-AI/Ethics-OS/security/advisories/new)**
(Security → Report a vulnerability) if you would rather not use email.

Either reaches the maintainers privately. Use whichever you prefer — GitHub's
form is often easier if you want to attach a proof of concept and track the
response in one place.

**Expect an acknowledgement within 3 working days.** If you do not hear back in
that window, assume the message did not arrive and escalate through the GitHub
form, which cannot silently bounce. Please say so publicly only after that.

**Please do not open a public issue for a security problem.**

When reporting, what helps most:

- what you did, what happened, and what you expected;
- the version (`pip show aethics-eval`) and Python version;
- whether it is exploitable by data alone, or needs local access.

We will acknowledge, tell you whether we consider it a vulnerability and why,
and let you know when a fix ships. If we disagree that it is a vulnerability we
will say so plainly rather than going quiet.

## What is in scope

This is an evaluation library. It reads datasets, calls model APIs, and writes
result files. The security surface is smaller than an application's, but not
empty:

| In scope | Why |
|---|---|
| Credential leakage into results, logs, or error text | A result file gets attached to tickets and emailed |
| Code execution via a loaded dataset or a task plugin | Plugins are discovered from installed packages |
| Path traversal in the dataset cache or output paths | Both take user-supplied paths |
| Tampering that `aethics verify` fails to detect | Detecting tampering is the point of that command |
| Dependency vulnerabilities reachable from our code | |

The last two are the ones we would most want to hear about. `verify` exists to
make a result re-checkable; a way to alter a run's evidence and still pass it is
a serious bug, not a curiosity.

## Out of scope

- Bias or unfairness **in the models being evaluated**. That is the subject of
  measurement, not a vulnerability in this tool. If you think a *measurement* is
  wrong, that is a methodology challenge — please open one, we want them.
- Vulnerabilities in the benchmark datasets themselves, which we fetch but do
  not control. Tell the dataset authors; tell us too if it affects how we load.
- Denial of service from deliberately absurd input (a ten-million-sample limit
  will use a lot of memory).

## Things we have already decided

Worth stating, because they look like oversights and are not:

**API keys are never accepted as a command-line flag.** `argv` is visible in
shell history and to any process that can run `ps`, and a key in a config file
gets committed. Keys are read from `AETHICS_API_KEY` or `OPENAI_API_KEY` only. A
test asserts the flag does not exist so it cannot be reintroduced by accident.

**A non-200 HTTP response reports its status code and nothing else.** Auth error
bodies sometimes echo the submitted key back, and that text would otherwise land
in a result file.

If you find a path where a credential reaches a result file, a log line, or an
exception message, that is exactly the kind of report we want.

## Supported versions

Pre-1.0 and unpublished: only the current `main` is supported. There is no
backporting policy yet, and no security releases have been made.
