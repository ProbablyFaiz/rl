"""Parses SLURM's nodelist shorthand, adapted from:
https://gist.github.com/ebirn/cf52876120648d7d85501fcbf185ff07?permalink_comment_id=5018013#gistcomment-5018013
"""


def _parse_int(s: str) -> tuple[str, str]:
    for i, c in enumerate(s):
        if c not in "0123456789":
            return s[:i], s[i:]
    return s, ""


def _parse_brackets(s: str) -> tuple[list[str], str]:
    # parse a "bracket" expression (including closing ']')
    lst = []
    while len(s) > 0:
        if s[0] == ",":
            s = s[1:]
            continue
        if s[0] == "]":
            return lst, s[1:]
        a, s = _parse_int(s)
        assert len(s) > 0, "Missing closing ']'"
        if s[0] in ",]":
            lst.append(a)
        elif s[0] == "-":
            b, s = _parse_int(s[1:])
            assert int(a) <= int(b), "Invalid range"
            # A leading zero on a lower boundary suggests that the
            # numerical part of the node name is padded with zeros,
            # e.g. nia0001.
            #
            # Just a single 0 on the lower boundary suggests a numerical
            # range without padding, e.g. nia[0-4].
            if a != "0" and a.startswith("0"):
                assert len(a) == len(b), (
                    "Boundaries of a ranged string with padding "
                    "must have the same length."
                )
                lst.extend([str(x).zfill(len(a)) for x in range(int(a), int(b) + 1)])
            elif a != "0" and b.startswith("0"):
                raise ValueError("Could not determine the padding style.")
            # If no padding is detected, simply use the range.
            else:
                lst.extend([str(x) for x in range(int(a), int(b) + 1)])
    assert len(s) > 0, "Missing closing ']'"


def _parse_node(s: str) -> tuple[list[str], str]:
    # parse a "node" expression
    for i, c in enumerate(s):
        if c == ",":  # name,...
            return [s[:i]], s[i + 1 :]
        if c == "[":  # name[v],...
            b, rest = _parse_brackets(s[i + 1 :])
            if len(rest) > 0:
                assert rest[0] == ",", f"Expected comma after brackets in {s[i:]}"
                rest = rest[1:]
            return [s[:i] + z for z in b], rest

    return [s], ""


def parse_nodes_str(nodes_str: str) -> list[str]:
    lst = []
    while len(nodes_str) > 0:
        v, nodes_str = _parse_node(nodes_str)
        lst.extend(v)
    return lst


if __name__ == "__main__":
    test_node_lists = [
        "clip-g1-[0-1],clip-g2-[2-3]",
        "clip-g1-0,clip-g2-0",
        "clip-g1-0,clip-g2-1",
        "clip-g1-1",
        "clip-a-[1,3,5]",
        "clip-b-[1-3,5]",
        "clip-c-[1-3,5,9-12]",
        "clip-d-[5,9-12]",
        "clip-e-[5,9],clip-e-[15-19]",
        "clip-f-[5,9],clip-f-[15,17]",
        "clip-f-5,clip-f-[15,17]",
        "clip-f-[5,9],clip-f-175",
        "cg[1-2]",
        "cg[001-002]",
        "cl[028,044,054]",
        "cg[001-002],cl[001-003,005-006,009-046]",
    ]
    for node_list in test_node_lists:
        print(parse_nodes_str(node_list))
