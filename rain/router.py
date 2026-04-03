"""
Rain ⛈️ — Keyword-Based Query Router

Classifies incoming queries and routes them to the most appropriate agent.
Rule-based keyword scoring — no extra model call, instant, fully offline.

Phase 6+: Combines routing with skill context injection and task decomposition.
"""

from typing import Optional
from .agents import AgentType


class AgentRouter:
    """
    Classifies incoming queries and routes them to the most appropriate agent.
    Rule-based — no extra model call, instant, fully offline.

    Scoring: keyword hits per category. Highest score wins.
    Tiebreaker: CODE > DOMAIN > REASONING > GENERAL

    Phase 6: skill context is injected separately by MultiAgentOrchestrator —
    routing still returns a core AgentType; skills augment, not replace, agents.
    """

    DOMAIN_KEYWORDS = [
        'bitcoin', 'lightning', 'satoshi', 'btc', 'blockchain', 'crypto',
        'sovereignty', 'sovereign', 'privacy', 'austrian', 'sound money',
        'sats', 'node', 'channel', 'wallet', 'utxo', 'taproot', 'segwit',
        'nostr', 'decentrali', 'self-custody', 'multisig', 'hodl', 'mining',
        'mempool', 'transaction', 'signature', 'schnorr', 'ecdsa', 'coinjoin',
        'lightning network', 'payment channel', 'bolt', 'bip', 'invoice',
    ]

    CODE_KEYWORDS = [
        'write', 'code', 'function', 'debug', 'implement', 'script',
        'program', 'fix', 'bug', 'class', 'algorithm', 'refactor',
        'build a', 'build the', 'build me', 'create a', 'make a', 'develop', 'api', 'library',
        'module', 'package', 'deploy', 'compile', 'syntax',
        'unit test', 'test case', 'test suite', 'test coverage', 'write tests', 'run tests',
        'error in', 'traceback', 'exception', 'import', 'install',
    ]

    REASONING_KEYWORDS = [
        'why', 'how does', 'explain', 'analyze', 'analyse', 'plan',
        'design', 'strategy', 'compare', 'difference', 'pros', 'cons',
        'tradeoff', 'should i', 'what is the best', 'recommend',
        'architecture', 'approach', 'think through', 'help me understand',
        'what would happen', 'evaluate', 'assess',
        # Abstract / philosophical question starters — these should always
        # route to LOGIC, never accidentally fall to DEV on a keyword tie.
        'how would', 'how would you', 'what makes', 'what should',
        'what is the', 'what are the', 'what is a', 'what is an',
        'most dangerous', 'most important', 'most common', 'most effective',
        'assumption', 'reliability', 'trade-off', 'implications', 'consequences',
        # Correction challenges — factual disputes are always a reasoning matter
        # and must route to LOGIC, not fall through to GENERAL.
        'actually', "you're wrong", "that's wrong", "that's incorrect",
        "that is wrong", "that is incorrect",
        # Quantitative / logical reasoning questions — these look like math or
        # logic puzzles but have no code keywords. They need the reasoning agent.
        'how much does', 'how much is', 'how much would', 'how much do',
        'how many', 'how many are', 'how many do',
        # Logical evaluation — "is X true/valid/correct/possible"
        'is it raining', 'is it possible', 'is it true', 'is it valid',
        'is this true', 'is this correct', 'is this valid', 'is this logical',
        'is this a', 'are all', 'are there', 'is there a reason',
        # Deductive reasoning starters
        'if it rains', 'if a then', 'if all', 'if every', 'if the',
        'therefore', 'thus', 'hence', 'conclude', 'deduce', 'infer',
        'logical', 'logically', 'fallacy', 'syllogism', 'deduction',
        'prove that', 'disprove', 'true or false', 'correct or incorrect',
        # Opinion / uncertainty questions — personal epistemic state
        'do you think', 'what do you think', 'do you believe',
        'something you believe', 'genuinely uncertain', 'are you certain',
        'what would you say', 'in your opinion', 'your view',
        # Logical puzzle / selection task phrases — never code
        'which must', 'must you', 'which card', 'which cards',
        'turn over', 'pick one', 'draw one',
        # Probability / statistics reasoning
        'probability', 'what is the probability', 'what are the odds',
        'base rate', 'false positive', 'true positive', 'bayes',
        'conditional', 'prior probability', 'posterior',
    ]

    # Phase 6: keywords that suggest the user wants Rain to *act*, not just answer.
    # A high task score (≥2) triggers the task-decomposition pipeline instead of
    # the normal reflect loop.  These are intentionally conservative — we only
    # enter task mode when the intent is unambiguous.
    TASK_KEYWORDS = [
        'refactor', 'migrate', 'set up', 'setup', 'automate',
        'step by step', 'implement a', 'develop a', 'build a system',
        'restructure', 'rewrite the', 'redesign', 'deploy', 'convert all',
        'create a script that', 'write a script that', 'make a tool that',
    ]

    # ReAct keywords — phrases that signal the answer requires *discovering* information
    # by inspecting the real world (filesystem, git, logs, running processes) rather
    # than reasoning about knowledge the model already has.
    # These are intentionally specific multi-word phrases to avoid false positives on
    # innocent uses of "find" or "check" in knowledge questions.
    REACT_KEYWORDS = [
        # Filesystem discovery
        'list files', 'list the files', 'list all files', 'what files', 'which files',
        'find files', 'find all files', 'show me the files', 'show the files',
        'read the file', 'open the file', 'show the contents', 'show contents of',
        "what's in the file", 'what is in the file', 'contents of the file',
        'look in the directory', 'look in the folder', 'what is in the directory',
        # Codebase search / discovery
        'find all', 'find where', 'where is the function', 'where is the class',
        'which file contains', 'which file has', 'what file has', 'what file contains',
        'grep for', 'search the codebase', 'find the function', 'find the class',
        'find the method', 'find all todo', 'find all fixme', 'find all occurrences',
        'find all instances', 'search for it in',
        # Git inspection
        'git log', 'git status', 'git diff', 'git branch',
        'recent commits', 'last commit', 'what was committed', 'commit history',
        'show the diff', 'what changed',
        # Log / output reading
        'check the log', 'read the log', 'show the log', 'tail the log',
        "what's in the log", 'what is in the log', 'look at the log',
        # Current system / process state
        "what's running", 'what is running', 'current state of', 'current status of',
        'is the server running', 'is the service', 'does the file exist',
        'is there a file', 'check if the file',
        # Verification via real execution
        'run and check', 'test whether it', 'verify that it works',
        'check if it works', 'does it work', 'is it working', 'run the tests',
    ]

    # Phase 7: keywords that indicate the user wants real-time / live information.
    # The hardcoded prefix '[web search results for:' is Rain's own injection marker —
    # it's an unambiguous signal that the message is already augmented with live data
    # and should be handled by the Search Agent rather than a general reasoning agent.
    SEARCH_KEYWORDS = [
        '[web search results for:',   # Rain's own search-augmented prefix — highest priority
        'current price', 'current fee', 'current rate', 'current version',
        'latest version', 'latest release', 'latest news', 'what is the latest',
        'right now', 'as of today', 'as of now', 'this week', 'this month',
        'what happened', 'who won', 'when did', 'when was',
        'live data', 'real-time', 'real time', 'up to date',
        'trending', 'breaking news', 'just released', 'just announced',
        'mempool fee', 'bitcoin price', 'btc price', 'fee rate',
        'exchange rate', 'market price', 'how much is btc',
    ]

    # Phrases that signal an advisory or meta question about Rain's own behavior.
    # These should always route to LOGIC regardless of keyword scoring — they are
    # asking HOW or WHAT, not asking Rain to DO something.
    META_QUESTION_SIGNALS = [
        'how should', 'what are the next steps', 'how to adjust', 'how to improve',
        'what does rain', 'how does rain', 'what should rain', 'why does rain',
        'how would rain', 'should rain', 'what is rain doing', 'how is rain',
        'what happens when rain', 'what happens after', 'after running',
        'after the fine', 'next steps after', 'steps after',
    ]

    def route(self, query: str) -> AgentType:
        """Classify query and return the most appropriate AgentType."""
        query_lower = query.lower().strip()

        # Meta/advisory questions about Rain's behavior or internals always
        # go to LOGIC — they are asking for analysis, not code execution.
        # Check this BEFORE keyword scoring to prevent false DEV routes.
        if any(sig in query_lower for sig in self.META_QUESTION_SIGNALS):
            return AgentType.LOGIC

        # Phase 7: Search Agent — highest priority check.
        # If the query is already augmented with web search results (Rain's own prefix),
        # route straight to the Search Agent — no scoring needed.
        # Also route when query strongly signals live/real-time intent.
        if query_lower.startswith('[web search results for:'):
            return AgentType.SEARCH
        search_score = sum(1 for kw in self.SEARCH_KEYWORDS if kw in query_lower)
        if search_score >= 2:
            # Guard against server boilerplate ("live data", "real-time numbers")
            # falsely triggering SEARCH on pure deduction/reasoning queries.
            # The server wraps augmented queries as:
            #   "...live data...real-time...\n\nQuestion: {original}"
            # Extract the original query and check for reasoning intent.
            _orig = query_lower
            if '\nquestion: ' in query_lower:
                _orig = query_lower.split('\nquestion: ', 1)[-1].strip()
            _reasoning_override = sum(1 for kw in self.REASONING_KEYWORDS if kw in _orig)
            if _reasoning_override >= 1:
                pass  # fall through — logic signal wins over search boilerplate
            else:
                return AgentType.SEARCH

        domain_score   = sum(1 for kw in self.DOMAIN_KEYWORDS   if kw in query_lower)
        code_score     = sum(1 for kw in self.CODE_KEYWORDS      if kw in query_lower)
        reasoning_score = sum(1 for kw in self.REASONING_KEYWORDS if kw in query_lower)

        # Boost code score if the query itself contains code
        if self._contains_code(query):
            code_score += 3

        # DEV requires clear plurality OR an explicit code imperative at the
        # start of the sentence.  A single ambiguous code keyword ('test',
        # 'function', 'class', 'algorithm') in a conceptual or design question
        # must NOT beat a reasoning signal — that's the mis-route we're fixing.
        #
        # _code_leading  — code score strictly exceeds both other scores
        # _code_explicit — query opens with a direct imperative code verb
        #                  ("write a ...", "implement ...", "debug this ...")
        #                  OR actual code syntax is present in the query body
        _CODE_IMPERATIVES = (
            'write ', 'code ', 'implement ', 'debug ', 'fix ', 'refactor ',
            'build ', 'create ', 'develop ', 'deploy ', 'compile ', 'generate ',
        )
        _code_leading  = code_score > max(domain_score, reasoning_score)
        _code_explicit = (
            self._contains_code(query)
            or any(query_lower.startswith(v) for v in _CODE_IMPERATIVES)
        )
        _code_wins = _code_leading or (code_score >= 1 and _code_explicit)

        best = max(code_score, domain_score, reasoning_score)
        if best == 0:
            # Short conversational exchanges (greetings, one-liners without a ?)
            # → GENERAL.  Everything else — actual questions, statements to
            # reason about — → LOGIC.  GENERAL should be a last resort, not the
            # default for anything we can't keyword-match.
            words = query_lower.split()
            is_question = '?' in query or len(words) > 7
            if is_question:
                return AgentType.LOGIC
            return AgentType.GENERAL
        if _code_wins and code_score == best:
            return AgentType.DEV
        if domain_score == best:
            return AgentType.DOMAIN
        if reasoning_score == best:
            return AgentType.LOGIC
        # Fallthrough: code matched but lost tie-breaking rules → reason about it
        return AgentType.LOGIC

    def should_use_react(self, query: str) -> bool:
        """
        Return True if the query requires the ReAct loop to discover information
        from the real world — filesystem, git, logs, running processes — rather
        than reasoning about knowledge the model already has.

        Unlike is_complex_task (which needs 2+ hits), a single clear REACT_KEYWORD
        phrase is enough because each phrase is already specific and multi-word.
        """
        q = query.lower()
        return any(kw in q for kw in self.REACT_KEYWORDS)

    def is_complex_task(self, query: str) -> bool:
        """
        Return True if the query looks like a multi-step task that should be
        decomposed and executed rather than answered in a single reflect loop.

        Threshold: 2+ task keywords OR a long query with at least 1 keyword.
        Intentionally conservative — single-step questions should never trigger
        task mode.
        """
        q = query.lower()
        hits = sum(1 for kw in self.TASK_KEYWORDS if kw in q)
        return hits >= 2 or (hits >= 1 and len(query.split()) > 40)

    def explain_mode(self, mode: str) -> str:
        """Human-readable label for the execution mode chosen by _auto_mode."""
        return {
            'react':   'ReAct loop (discovery query detected)',
            'reflect': 'recursive reflect',
        }.get(mode, mode)

    def _contains_code(self, text: str) -> bool:
        """Check if the query itself contains code, not just asks about it."""
        code_starters = ('def ', 'class ', 'import ', 'function ', 'const ', '```',
                         'async ', '@', '#include', '#!/')
        return any(text.strip().startswith(s) or f'\n{s}' in text for s in code_starters)

    def explain(self, agent_type: AgentType) -> str:
        """Human-readable explanation of why this agent was chosen."""
        labels = {
            AgentType.DEV:        "Dev Agent (code task detected)",
            AgentType.LOGIC:      "Logic Agent (reasoning task detected)",
            AgentType.DOMAIN:     "Domain Expert (Bitcoin/sovereignty topic detected)",
            AgentType.GENERAL:    "General Agent",
            AgentType.REFLECTION: "Reflection Agent",
            AgentType.SYNTHESIZER:"Synthesizer",
            AgentType.SEARCH:     "Search Agent (live web results)",
            AgentType.TASK:       "Task Agent (autonomous execution)",
        }
        return labels.get(agent_type, agent_type.value)
