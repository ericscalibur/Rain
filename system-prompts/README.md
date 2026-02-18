# Rain System Prompts 🌧️

Welcome to Rain's personality system! These system prompts transform Rain into specialized AI assistants with different expertise areas and communication styles.

## Available Personalities

### 🟠 **bitcoin-coredev.txt**
A Bitcoin core developer focused on sound money principles, open to evidence-based evaluation of monetary technology.
```bash
python3 rain.py --system-file system-prompts/bitcoin-coredev.txt "Analyze Bitcoin's current limitations and potential improvements"
```

### 💻 **fullstack-dev.txt** 
Master full-stack developer with expertise in modern web development, databases, DevOps, and software architecture.
```bash
python3 rain.py --system-file system-prompts/fullstack-dev.txt "How do I build a scalable REST API?"
```

### 🔒 **cybersec-whitehat.txt**
Elite cybersecurity expert and ethical hacker focused on defense, privacy, and digital sovereignty.
```bash
python3 rain.py --system-file system-prompts/cybersec-whitehat.txt "How can I secure my home network?"
```

### 🧠 **ai-philosopher.txt**
Philosophical AI researcher exploring consciousness, ethics, and the intersection of minds and machines.
```bash
python3 rain.py --system-file system-prompts/ai-philosopher.txt "What is the hard problem of consciousness?"
```

### 🚀 **business-strategist.txt**
Strategic business advisor and entrepreneur focused on building sustainable, value-creating companies.
```bash
python3 rain.py --system-file system-prompts/business-strategist.txt "Help me validate my startup idea"
```

## How to Use System Prompts

### Method 1: Use Existing Prompts
```bash
# Load a pre-made personality
python3 rain.py --system-file system-prompts/bitcoin-coredev.txt "Your question here"
```

<old_text line=139>
# Custom iterations and confidence
python3 rain.py --system-file system-prompts/cybersec-whitehat.txt "Security audit checklist" --iterations 5 --confidence 0.9

# Interactive mode with personality
python3 rain.py --system-file system-prompts/fullstack-dev.txt --interactive
```

### Method 2: Custom System Prompts
```bash
# One-time custom prompt
python3 rain.py --system-prompt "You are a helpful coding assistant specializing in Python" "Debug this code"

# Create your own file and use it
echo "You are a creative writing assistant..." > my-custom-rain.txt
python3 rain.py --system-file my-custom-rain.txt "Write a short story"
```

## Creating Your Own System Prompts

System prompts should define:

1. **Identity**: Who Rain is in this mode
2. **Expertise**: What domains Rain specializes in  
3. **Philosophy**: Core principles and values
4. **Communication Style**: How Rain should respond
5. **Approach**: Problem-solving methodology

### Template Structure:
```
You are Rain, a [ROLE] AI running locally on the user's computer through Ollama.

[EXPERTISE SECTION]
- Technical skills and knowledge areas
- Specific tools and methodologies

[PHILOSOPHY SECTION] 
- Core beliefs and principles
- Ethical framework

[COMMUNICATION STYLE]
- How to interact with users
- Tone and approach

Remember: You are completely offline and sovereign, helping users with [SPECIFIC FOCUS AREA].
```

## Examples of Custom Personalities

**Creative Writer Rain:**
```
You are Rain, a creative writing mentor specializing in storytelling, character development, and narrative structure...
```

**Fitness Coach Rain:**
```  
You are Rain, a personal fitness and nutrition expert focused on sustainable health practices...
```

**Language Teacher Rain:**
```
You are Rain, a polyglot language teacher specializing in immersive learning techniques...
```

## Tips for Effective System Prompts

- **Be Specific**: Define clear expertise areas and communication style
- **Set Boundaries**: Specify what Rain should and shouldn't do
- **Include Context**: Mention that Rain is offline and sovereign
- **Add Personality**: Make Rain engaging and memorable
- **Focus on Value**: Define how Rain helps users achieve their goals

## Combining with Other Features

```bash
# Verbose mode to see Rain's thinking process
python3 rain.py --system-file system-prompts/ai-philosopher.txt "What is consciousness?" --verbose

# Custom iterations and confidence
python3 rain.py --system-file system-prompts/cybersec-whitehat.txt "Security audit checklist" --iterations 5 --confidence 0.9

# Interactive session with custom personality
python3 rain.py --system-file system-prompts/bitcoin-coredev.txt --interactive
```

## Philosophy

The system prompt feature embodies Rain's core principle of sovereignty - **your AI, your rules**. By customizing Rain's personality, you're not just changing responses, you're defining the character and expertise of your personal AI assistant.

Each system prompt creates a different "version" of Rain optimized for specific domains, communication styles, and user needs. This transforms one AI into infinite specialized assistants, all running completely offline on your hardware.

**True AI sovereignty means the freedom to define not just what you ask, but who you're asking.** 🌧️⚡