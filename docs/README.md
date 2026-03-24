Plugin-based Input Sources and CI

Overview
- This project introduces a plugin-based input source framework for loading experiment data.
- Input sources are pluggable and discoverable at runtime via a central registry.
- The ecosystem includes a Spots input, as well as Kafka/RabbitMQ/Filesystem/Zip sources, with tests and CI wiring.

Directory structure (high level)
- input_sources/      # input plugins and registry
- experiment_analysis/  # pipeline and processors
- image/               # image processing helpers
- tests/               # test suite
- docs/                # documentation and examples (this folder)
- .github/workflows/  # CI configuration

Examples (see docs/EXAMPLES_INPUT_PLUGINS.md for details)
- See docs/EXAMPLES_INPUT_PLUGINS.md

Docs
- docs/EXAMPLES_INPUT_PLUGINS.md: practical usage examples for input plugins
- docs/README.md: this file
