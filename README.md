# Rain ⛈️ - Sovereign AI Ecosystem

*"Be like rain - essential, unstoppable, and free."*

## Mission Statement

Rain is a sovereign AI ecosystem designed to bring true digital independence through recursive small language models (SLMs). Just as rain falls naturally without permission, fees, or restrictions, Rain provides AI capabilities that are:

- **Sovereign** - Completely owned and controlled by you
- **Private** - Runs entirely offline on your hardware  
- **Free** - No subscriptions, APIs, or dependencies
- **Recursive** - Multiple specialized models working together
- **Self-Improving** - Gets smarter through iterative reflection

Rain represents the convergence of sovereign money (Bitcoin/Lightning) and sovereign AI - the foundational technologies for true digital freedom.

## Philosophy

We believe AI should be:
- **Owned, not rented** - Your AI, your rules, your data
- **Decentralized** - No single points of failure or control
- **Accessible** - Runs on consumer hardware, not datacenters
- **Transparent** - Open source, auditable, modifiable
- **Collaborative** - Multiple models working together > one massive model

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              Rain Orchestrator          │
├─────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐ │
│  │Dev Agent│ │Logic    │ │Domain       │ │
│  │(7B SLM) │ │Agent    │ │Expert       │ │
│  │         │ │(7B SLM) │ │(7B SLM)     │ │
│  └─────────┘ └─────────┘ └─────────────┘ │
├─────────────────────────────────────────┤
│         Recursive Reflection Layer      │
├─────────────────────────────────────────┤
│              Ollama Runtime             │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Specialized Agents (7B Models)
- **Dev Agent**: Code generation, debugging, refactoring
- **Logic Agent**: Reasoning, planning, problem decomposition  
- **Domain Expert**: Specialized knowledge (Coding, Encryption,Bitcoin, Lightning, etc.)
- **Reflection Agent**: Quality control and iterative improvement

### 2. Orchestrator
- Routes queries to appropriate agents
- Manages recursive reflection cycles
- Synthesizes multi-agent responses
- Handles consensus building

### 3. Recursive Reflection System
- Models review and critique their own outputs
- Iterative refinement until consensus or max iterations
- Self-improvement through reflection loops
- Quality gates and validation

## Technical Requirements

### Hardware
- **Minimum**: 16GB RAM, 4-core CPU
- **Recommended**: 32GB RAM, 8-core CPU
- **Storage**: 50GB+ for model storage
- **Network**: None required (fully offline)

### Software Stack
- **Runtime**: Ollama (local model management)
- **Models**: Multiple 7B parameter models
- **Language**: Python/Node.js orchestration layer
- **Interface**: CLI, API, and web interface

## Development Roadmap

### Phase 1: Foundation (MVP)
- [ ] Set up Ollama with base 7B model
- [ ] Create simple orchestrator
- [ ] Implement basic recursive reflection
- [ ] CLI interface for interaction

### Phase 2: Multi-Agent System
- [ ] Deploy specialized agent models
- [ ] Build agent routing logic
- [ ] Create consensus mechanisms
- [ ] Add quality validation

### Phase 3: Self-Improvement
- [ ] Implement reflection loops
- [ ] Add model fine-tuning capabilities
- [ ] Create feedback learning systems
- [ ] Build performance metrics

### Phase 4: Advanced Features
- [ ] Web interface
- [ ] API endpoints
- [ ] Plugin system
- [ ] Model marketplace

## Getting Started

```bash
# Clone the repository
git clone https://github.com/ericscalibur/Rain.git
cd Rain

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download base models
ollama pull llama3.1:7b
ollama pull codellama:7b

# Run Rain
python3 rain.py "Your question here"
```

## System Prompts - Customize Rain's Personality

Rain's most powerful feature is its ability to transform into specialized AI experts through system prompts. Instead of one generic AI, you get infinite customized personalities.

### Available Personalities

```bash
# Bitcoin maximalist focused on sound money and Austrian economics
python3 rain.py --system-file system-prompts/bitcoin-maximalist.txt --interactive

# Master full-stack developer with modern web expertise  
python3 rain.py --system-file system-prompts/fullstack-dev.txt --interactive

# Cybersecurity expert and ethical hacker
python3 rain.py --system-file system-prompts/cybersec-whitehat.txt --interactive

# AI philosopher exploring consciousness and ethics
python3 rain.py --system-file system-prompts/ai-philosopher.txt --interactive

# Business strategist and entrepreneur advisor
python3 rain.py --system-file system-prompts/business-strategist.txt --interactive
```

### Custom System Prompts

Create your own AI personalities:

```bash
# Use a custom system prompt directly
python3 rain.py --system-prompt "You are a creative writing mentor..." --interactive

# Create and save your own personality file
echo "You are Rain, a [YOUR SPECIALTY] expert..." > my-rain-personality.txt
python3 rain.py --system-file my-rain-personality.txt --interactive
```

See the `system-prompts/` directory for examples and templates to create your own specialized Rain personalities.

## Usage Examples

### Basic Usage
```bash
# Single question
python3 rain.py "Explain quantum computing"

# Interactive chat mode
python3 rain.py --interactive

# See Rain's thinking process
python3 rain.py "Complex question" --verbose
```

### Advanced Features
```bash
# Custom reflection settings
python3 rain.py --iterations 5 --confidence 0.9 "Hard problem"

# Combine personality with custom settings
python3 rain.py --system-file system-prompts/ai-philosopher.txt "What is consciousness?" --verbose
```

## Why "Rain"?

Rain symbolizes everything we believe AI should be:
- **Life-giving** - Nurtures growth and possibility
- **Sovereign** - Falls freely, uncontrolled by any authority  
- **Essential** - Fundamental to life and progress
- **Natural** - Flows where it's needed most
- **Renewable** - Infinite, sustainable resource

## Contributing

Rain is built on the principle of collective sovereignty. We welcome contributors who share our vision of AI freedom and independence.

## License

MIT License - Because freedom should be free.

---

*"Just as rain brings life from the clouds to the earth, Rain brings AI from the datacenter to your laptop."*

⛈️ **Rain: Your AI, Your Rules, Your Future** ⛈️
