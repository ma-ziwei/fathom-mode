# Changelog

## [0.1.0] — 2026-03-18

First public release.

### Features
- Multi-round intent elicitation — system asks targeted questions to understand what you actually want before generating
- Every round provides a high-quality response AND a follow-up question — useful output from round 1, not just at the end
- Structured prompt compilation — compiles your intent into an auditable, structured prompt for any downstream LLM
- Causal reasoning protection — your stated cause-effect relationships are preserved, never silently replaced by LLM correlation
- Fathom Score — real-time depth indicator showing how deeply your intent has been understood
- User-controlled compilation — say "fathom" when you're ready, the system never decides for you
- 5 collaboration modes — thinking, creation, execution, learning, general
- Provider support — OpenAI, Anthropic, Gemini, DeepSeek, OpenClaw, or bring your own LLM
- CLI with session persistence
- OpenClaw skill for instant agent integration
- Cognitive bias detection (8 bias types with correction guidance)
