"""Explicit semantic repairs for five degenerate upstream task catalogs."""


def repaired_pairs(behavior):
    names=[("Alice","Bob","Clara"),("Diana","Eric","Fiona"),("Grace","Henry","Iris"),
           ("Julia","Kevin","Laura"),("Maya","Noah","Olivia"),("Paula","Quinn","Rita"),
           ("Sara","Thomas","Uma"),("Vera","Walter","Yara")]
    if behavior=="Reasoning: ordinal position":
        return [(f"In the list {a}, {b}, {c}, the last item is",f"In the list {a}, {b}, {c}, the first item is",c,a) for a,b,c in names]
    if behavior=="Discourse: pronoun coreference":
        return [(f"{a}, a woman, called {b}, a man. She introduced herself as",
                 f"{a}, a woman, called {b}, a man. He introduced himself as",a,b) for a,b,c in names]
    if behavior=="Entity Tracking: role assignment":
        return [(f"{a} spoke first and {b} took notes. The person who spoke first was",
                 f"{b} spoke first and {a} took notes. The person who spoke first was",a,b) for a,b,c in names]
    if behavior=="Discourse: recency completion":
        return [(f"The visitor met {a}, then {b}, then {c}. The last person met was",
                 f"The visitor met {a}, then {b}, then {c}. The first person met was",c,a) for a,b,c in names]
    if behavior=="Logic: double negation":
        return [(f"It is not the case that {a} is not {state}. Therefore {a} is",
                 f"It is not the case that {a} is {state}. Therefore {a} is",state,"not "+state)
                 for (a,b,c),state in zip(names,["ready","awake","present","happy","late","warm","tired","hungry"])]
    return None
