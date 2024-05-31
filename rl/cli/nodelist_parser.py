"""Parses SLURM's nodelist shorthand, adapted from:
https://gist.github.com/ebirn/cf52876120648d7d85501fcbf185ff07?permalink_comment_id=5018013#gistcomment-5018013
"""


def _parse_int(s: str) -> tuple[str, str]:
    for i, c in enumerate(s):
        if c not in "0123456789":
            return s[:i], s[i:]
    return s, ""


def _parse_brackets(s: str) -> tuple[list[str], str]:
    lst: list[str] = []
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
    return lst, s[1:]


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
        "sh02-01n[61-62,65,67,69,71],sh02-02n[02,09,12,25-31,34,37,39-44,46,49-50,54,58-59,62,65-67,71-72],sh02-03n[18,21,25-27,29-35,47,50,52-53,56,58-60,62-72],sh02-04n[05,18,21-23,25-26,28-29,31,33,35-36,39,45,49-50,53,55-58,61,63-64,67,69-70],sh02-05n[01-05,07,09-10,15-16,21-24,31,48,52,69],sh02-06n[01-03,13-14,16,18,20-29,31-32,34-35,37-52,56-59,61,64-69,71],sh02-07n[01,03,05-07,09,14-15,19,21-22,24,29-34,36-37,39-43,45-58,64-72],sh02-08n[02-05,07-10,12-20,29,31-36,59-72],sh02-09n[01-08,11-12,14-44,49-57,59-63,65-68,70-72],sh02-10n[05-09,11,14,16-20,24,35-40,49-52,55-61,66-67],sh02-12n[03,07,09-11,13-17],sh02-13n[02-05,11],sh02-14n[07,11,14-15],sh02-15n[01-02,07],sh03-03n[04,16,20,27,29,32,35-36,41,57-58,65],sh03-04n[01-02,07,09,14-16,21-27,29-35,37-41,43-47,54-56,61-63,68-72],sh03-06n[02-20,22-24,37-52,57-65,67,69-72],sh03-07n[03,10-16,19,27,31,33,36-37,42,45-48,50-51,53],sh03-08n[06-10,12,14-16,18-24,50-52,55-59,61-67,69,71-72],sh03-09n[01-03,05-18,20,22-23,26,32,35,39-40,42,44-49,51,54,56-59,61,63-65,68-70,72],sh03-10n[01-03,05-08,10-12,14-20,22-26,28,33-34,36,40,43-51,58-61,64,69-72],sh03-11n[02-08,11,13-15,17,19-21],sh03-12n[20-24],sh03-13n[03-09,11,13,16,18,20-23],sh03-14n[01-03,07,09,11,13,15,17-24],sh03-15n[01-05,07-24],sh03-16n[01-04,07-09,11,13-19,21-24],sh03-17n[01,03,05,10-13,17,20-22,24]",
    ]
    for node_list in test_node_lists:
        print(parse_nodes_str(node_list))
