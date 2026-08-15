# Security policy

Please do not publish a suspected security issue as a public issue before the
maintainers have had a chance to assess it. Until a project contact is listed,
report it privately to the repository owner.

This project parses untrusted model files only for local development. Treat
third-party checkpoints and tokenizer files as untrusted input, and keep the
loader's bounds and schema checks intact when contributing changes.
