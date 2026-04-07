from __future__ import annotations
"""DFA Regex — compile regex to DFA for O(n) guaranteed matching.
No backtracking. Used for REGEXP_MATCH acceleration.

Thompson NFA construction → subset construction → DFA.
Only supports: literal chars, '.', '*', '+', '?', '|', '(', ')',
character classes [abc], [a-z], [^abc]."""
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


import threading

_NFA_COUNTER_LOCK = threading.Lock()


class _NFAState:
    __slots__ = ('id', 'transitions', 'epsilon')
    _counter = 0

    def __init__(self) -> None:
        with _NFA_COUNTER_LOCK:
            self.id = _NFAState._counter
            _NFAState._counter += 1
        self.transitions: Dict[str, List[_NFAState]] = {}
        self.epsilon: List[_NFAState] = []


class _NFA:
    __slots__ = ('start', 'accept')

    def __init__(self, start: _NFAState, accept: _NFAState) -> None:
        self.start = start
        self.accept = accept


def _char_nfa(ch: str) -> _NFA:
    s, a = _NFAState(), _NFAState()
    s.transitions[ch] = [a]
    return _NFA(s, a)


def _dot_nfa() -> _NFA:
    """Match any character."""
    s, a = _NFAState(), _NFAState()
    s.transitions['.'] = [a]
    return _NFA(s, a)


def _concat_nfa(a: _NFA, b: _NFA) -> _NFA:
    a.accept.epsilon.append(b.start)
    return _NFA(a.start, b.accept)


def _union_nfa(a: _NFA, b: _NFA) -> _NFA:
    s, f = _NFAState(), _NFAState()
    s.epsilon.extend([a.start, b.start])
    a.accept.epsilon.append(f)
    b.accept.epsilon.append(f)
    return _NFA(s, f)


def _star_nfa(a: _NFA) -> _NFA:
    s, f = _NFAState(), _NFAState()
    s.epsilon.extend([a.start, f])
    a.accept.epsilon.extend([a.start, f])
    return _NFA(s, f)


def _plus_nfa(a: _NFA) -> _NFA:
    s, f = _NFAState(), _NFAState()
    s.epsilon.append(a.start)
    a.accept.epsilon.extend([a.start, f])
    return _NFA(s, f)


def _question_nfa(a: _NFA) -> _NFA:
    s, f = _NFAState(), _NFAState()
    s.epsilon.extend([a.start, f])
    a.accept.epsilon.append(f)
    return _NFA(s, f)


def _epsilon_closure(states: Set[_NFAState]) -> FrozenSet[int]:
    """Compute epsilon closure of a set of NFA states."""
    stack = list(states)
    closure: Set[int] = set()
    visited: Set[int] = set()
    while stack:
        st = stack.pop()
        if st.id in visited:
            continue
        visited.add(st.id)
        closure.add(st.id)
        for eps in st.epsilon:
            if eps.id not in visited:
                stack.append(eps)
    return frozenset(closure)


def _epsilon_closure_states(states: Set[_NFAState]) -> Set[_NFAState]:
    stack = list(states)
    result: Set[_NFAState] = set()
    visited: Set[int] = set()
    while stack:
        st = stack.pop()
        if st.id in visited:
            continue
        visited.add(st.id)
        result.add(st)
        for eps in st.epsilon:
            if eps.id not in visited:
                stack.append(eps)
    return result


class DFARegex:
    """Compiled DFA regex matcher. O(n) matching guaranteed."""

    __slots__ = ('_transitions', '_accept_states', '_start', '_num_states')

    def __init__(self, transitions: Dict[int, Dict[str, int]],
                 accept_states: Set[int], start: int, num_states: int) -> None:
        self._transitions = transitions
        self._accept_states = accept_states
        self._start = start
        self._num_states = num_states

    @staticmethod
    def compile(pattern: str) -> DFARegex:
        """编译正则为 DFA。"""
        with _NFA_COUNTER_LOCK:
            _NFAState._counter = 0
        nfa = _parse_regex(pattern)
        return _nfa_to_dfa(nfa)

    def match(self, text: str) -> bool:
        """O(n) full match — does the entire text match the pattern?"""
        state = self._start
        for ch in text:
            trans = self._transitions.get(state, {})
            if ch in trans:
                state = trans[ch]
            elif '.' in trans:
                state = trans['.']
            else:
                return False
        return state in self._accept_states

    def search(self, text: str) -> bool:
        """Does the pattern occur anywhere in text?"""
        for start in range(len(text)):
            state = self._start
            for i in range(start, len(text)):
                trans = self._transitions.get(state, {})
                if text[i] in trans:
                    state = trans[text[i]]
                elif '.' in trans:
                    state = trans['.']
                else:
                    break
                if state in self._accept_states:
                    return True
            # Check empty match
            if self._start in self._accept_states:
                return True
        return False

    def find_all(self, text: str) -> List[Tuple[int, int]]:
        """Find all non-overlapping matches. Returns [(start, end), ...]."""
        results: List[Tuple[int, int]] = []
        i = 0
        while i < len(text):
            state = self._start
            last_match = -1
            if state in self._accept_states:
                last_match = i
            for j in range(i, len(text)):
                trans = self._transitions.get(state, {})
                if text[j] in trans:
                    state = trans[text[j]]
                elif '.' in trans:
                    state = trans['.']
                else:
                    break
                if state in self._accept_states:
                    last_match = j + 1
            if last_match > i:
                results.append((i, last_match))
                i = last_match
            else:
                i += 1
        return results


def _parse_regex(pattern: str) -> _NFA:
    """Parse regex pattern into NFA using Thompson construction."""
    pos = [0]

    def parse_expr() -> _NFA:
        left = parse_concat()
        while pos[0] < len(pattern) and pattern[pos[0]] == '|':
            pos[0] += 1
            right = parse_concat()
            left = _union_nfa(left, right)
        return left

    def parse_concat() -> _NFA:
        parts: List[_NFA] = []
        while pos[0] < len(pattern) and pattern[pos[0]] not in ('|', ')'):
            parts.append(parse_quantifier())
        if not parts:
            s, a = _NFAState(), _NFAState()
            s.epsilon.append(a)
            return _NFA(s, a)
        result = parts[0]
        for i in range(1, len(parts)):
            result = _concat_nfa(result, parts[i])
        return result

    def parse_quantifier() -> _NFA:
        base = parse_atom()
        if pos[0] < len(pattern):
            if pattern[pos[0]] == '*':
                pos[0] += 1; return _star_nfa(base)
            if pattern[pos[0]] == '+':
                pos[0] += 1; return _plus_nfa(base)
            if pattern[pos[0]] == '?':
                pos[0] += 1; return _question_nfa(base)
        return base

    def parse_atom() -> _NFA:
        if pos[0] >= len(pattern):
            s, a = _NFAState(), _NFAState()
            s.epsilon.append(a)
            return _NFA(s, a)
        ch = pattern[pos[0]]
        if ch == '(':
            pos[0] += 1
            nfa = parse_expr()
            if pos[0] < len(pattern) and pattern[pos[0]] == ')':
                pos[0] += 1
            return nfa
        if ch == '.':
            pos[0] += 1
            return _dot_nfa()
        if ch == '\\' and pos[0] + 1 < len(pattern):
            pos[0] += 1
            escaped = pattern[pos[0]]
            pos[0] += 1
            if escaped == 'd':
                return _char_class_nfa('0123456789')
            if escaped == 'w':
                chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'
                return _char_class_nfa(chars)
            if escaped == 's':
                return _char_class_nfa(' \t\n\r')
            return _char_nfa(escaped)
        if ch == '[':
            return parse_char_class()
        if ch in ('*', '+', '?', '|', ')'):
            s, a = _NFAState(), _NFAState()
            s.epsilon.append(a)
            return _NFA(s, a)
        pos[0] += 1
        return _char_nfa(ch)

    def parse_char_class() -> _NFA:
        pos[0] += 1  # skip '['
        negate = False
        if pos[0] < len(pattern) and pattern[pos[0]] == '^':
            negate = True
            pos[0] += 1
        chars: List[str] = []
        while pos[0] < len(pattern) and pattern[pos[0]] != ']':
            if (pos[0] + 2 < len(pattern) and pattern[pos[0] + 1] == '-'
                    and pattern[pos[0] + 2] != ']'):
                start_ch = pattern[pos[0]]
                end_ch = pattern[pos[0] + 2]
                for c in range(ord(start_ch), ord(end_ch) + 1):
                    chars.append(chr(c))
                pos[0] += 3
            else:
                chars.append(pattern[pos[0]])
                pos[0] += 1
        if pos[0] < len(pattern):
            pos[0] += 1  # skip ']'
        if negate:
            all_printable = [chr(c) for c in range(32, 127)]
            chars = [c for c in all_printable if c not in chars]
        return _char_class_nfa(''.join(chars))

    return parse_expr()


def _char_class_nfa(chars: str) -> _NFA:
    """Create NFA matching any character in the class."""
    s, a = _NFAState(), _NFAState()
    for ch in chars:
        s.transitions.setdefault(ch, []).append(a)
    return _NFA(s, a)


def _nfa_to_dfa(nfa: _NFA) -> DFARegex:
    """Subset construction: NFA → DFA."""
    all_nfa_states: Dict[int, _NFAState] = {}

    def collect(st: _NFAState) -> None:
        if st.id in all_nfa_states:
            return
        all_nfa_states[st.id] = st
        for targets in st.transitions.values():
            for t in targets:
                collect(t)
        for e in st.epsilon:
            collect(e)

    collect(nfa.start)
    accept_id = nfa.accept.id

    # Compute alphabet
    alphabet: Set[str] = set()
    for st in all_nfa_states.values():
        alphabet.update(st.transitions.keys())

    # Subset construction
    start_closure = _epsilon_closure_states({nfa.start})
    start_key = frozenset(s.id for s in start_closure)

    dfa_states: Dict[FrozenSet[int], int] = {start_key: 0}
    dfa_transitions: Dict[int, Dict[str, int]] = {}
    dfa_accept: Set[int] = set()
    queue: List[Tuple[FrozenSet[int], Set[_NFAState]]] = [(start_key, start_closure)]
    counter = 1

    if accept_id in start_key:
        dfa_accept.add(0)

    while queue:
        state_key, state_set = queue.pop(0)
        dfa_id = dfa_states[state_key]
        dfa_transitions[dfa_id] = {}

        for ch in alphabet:
            # Find all states reachable via ch
            next_states: Set[_NFAState] = set()
            for nfa_st in state_set:
                if ch in nfa_st.transitions:
                    next_states.update(nfa_st.transitions[ch])
            if not next_states:
                continue
            closure = _epsilon_closure_states(next_states)
            closure_key = frozenset(s.id for s in closure)

            if closure_key not in dfa_states:
                dfa_states[closure_key] = counter
                counter += 1
                queue.append((closure_key, closure))
                if accept_id in closure_key:
                    dfa_accept.add(dfa_states[closure_key])

            dfa_transitions[dfa_id][ch] = dfa_states[closure_key]

    return DFARegex(dfa_transitions, dfa_accept, 0, counter)
