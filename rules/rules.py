# Define outpoint and pluspoint descriptions
outpoint_descriptions = {
    "omitted_data": "An omitted anything is an outpoint. This can be an omitted person, terminal, object, energy, space, time, form, sequence, or even an omitted scene.",
    "altered_sequence": "Any things, events, objects, sizes in a wrong sequence is an outpoint.",
    "dropped_time": "Time that should be noted and isn't would be an outpoint of dropped time. It is a special case of an omitted datum.",
    "falsehood": "When you hear two facts that are contrary, one is a falsehood or both are. A false anything qualifies for this outpoint.",
    "altered_importance": "An importance shifted from its actual relative importance, up or down. An outpoint.",
    "wrong_target": "Mistaken objective wherein one believes he is or should be reaching toward A and finds he is or should be reaching toward B is an outpoint.",
    "wrong_source": "Information taken from wrong source, orders taken from the wrong source, gifts or materiel taken from wrong source all add up to eventual confusion and possible trouble.",
    "contrary_facts": "When two statements are made on one subject which are contrary to each other, we have contrary facts.",
    "added_time": "In this outpoint we have the reverse of dropped time. In added time we have, as the most common example, something taking longer than it possibly could.",
    "added_inapplicable_data": "Just plain added data does not necessarily constitute an outpoint. It may be someone being thorough. But when the data is in no way applicable to the scene or situation and is added it is a definite outpoint.",
    "incorrectly_included_datum": "A part from one class of parts is included wrongly in another class of parts. So there is an incorrectly included datum which is a companion to the omitted datum as an outpoint.",
    "assumed_identities_not_identical": "This outpoint occurs when things that are actually different are treated as if they're identical.",
    "assumed_similarities_not_similar": "This outpoint occurs when things that don't share meaningful characteristics are treated as if they're similar.",
    "assumed_differences_not_different": "This outpoint occurs when things that are actually identical or the same class are treated as if they're different."
}

pluspoint_descriptions = {
    "related_facts_known": "All relevant facts known.",
    "events_in_correct_sequence": "Events in actual sequence.",
    "time_noted": "Time is properly noted.",
    "data_proven_factual": "Data must be factual, which is to say, true and valid.",
    "correct_relative_importance": "The important and unimportant are correctly sorted out.",
    "expected_time_period": "Events occurring or done in the time one would reasonably expect them to be.",
    "adequate_data": "No sectors of omitted data that would influence the situation.",
    "applicable_data": "The data presented or available applies to the matter in hand and not something else.",
    "correct_source": "Not wrong source.",
    "correct_target": "Not going in some direction that would be wrong for the situation.",
    "data_in_same_classification": "Data from two or more different classes of material not introduced as the same class.",
    "identities_are_identical": "Not similar or different.",
    "similarities_are_similar": "Things that share meaningful characteristics are recognized as similar.",
    "differences_are_different": "Things that are actually different are recognized as different."
}


class Rule:
    """Base class for outpoint and pluspoint rules."""

    def __init__(self, name, description):
        """Initialize a rule with a name and description."""
        self.name = name
        self.description = description

    def check(self, statement, truth_graph):
        """Check if the rule applies to the statement."""
        raise NotImplementedError("Subclasses must implement this method")


class OutpointRule(Rule):
    """Rule for detecting logical errors (outpoints) in statements."""

    def __init__(self, name):
        """Initialize an outpoint rule."""
        super().__init__(name, outpoint_descriptions.get(
            name, f"Outpoint: {name}"))

    def check(self, statement, truth_graph):
        """Check if the statement exhibits this outpoint."""
        # Basic implementation - in a real system, this would have specific logic for each rule
        # For now, we'll rely on the LLM evaluation
        return False


class PluspointRule(Rule):
    """Rule for detecting logical strengths (pluspoints) in statements."""

    def __init__(self, name):
        """Initialize a pluspoint rule."""
        super().__init__(name, pluspoint_descriptions.get(
            name, f"Pluspoint: {name}"))

    def check(self, statement, truth_graph):
        """Check if the statement exhibits this pluspoint."""
        # Basic implementation - in a real system, this would have specific logic for each rule
        # For now, we'll rely on the LLM evaluation
        return False


# Create rule instances
RULES_OUT = [OutpointRule(name) for name in outpoint_descriptions.keys()]
RULES_PLUS = [PluspointRule(name) for name in pluspoint_descriptions.keys()]
