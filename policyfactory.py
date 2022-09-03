from newtestpolicy1 import newtestpolicy1

def none_policy():
    return None


policy_factory = dict()
policy_factory['newtestpolicy1'] = newtestpolicy1  # can be customized based on the need.
policy_factory['none'] = none_policy