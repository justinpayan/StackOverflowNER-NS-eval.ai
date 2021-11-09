import json
import os
import random
import re

import numpy as np


def evaluate(test_annotation_file, user_annotation_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_annotation_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    # Load in the 2 json files, use conlleval script.
    with open(test_annotation_file, 'rb') as f:
        test_annots = json.load(f)

    with open(user_annotation_file, 'rb') as f:
        user_annots = json.load(f)

    assert len(user_annots) == 5
    for i in range(1, 6):
        ep_name = "episode%d" % i
        assert ep_name in user_annots
        user_ids = set([x["id"] for x in user_annots[ep_name]])
        test_ids = set([x["id"] for x in test_annots[ep_name]])
        assert user_ids == test_ids

    user_map = {}
    test_map = {}
    for i in range(1, 6):
        ep_name = "episode%d" % i
        user_map[ep_name] = {}
        test_map[ep_name] = {}
        for x in user_annots[ep_name]:
            user_map[ep_name][x["id"]] = {"sentence": x["sentence"], "tag_sequence": x["tag_sequence"]}
        for x in test_annots[ep_name]:
            test_map[ep_name][x["id"]] = {"sentence": x["sentence"], "tag_sequence": x["tag_sequence"]}

    # Collect the fscores across all 5 test episodes.
    fscores = []
    for i in range(1, 6):
        print("evaluating episode %d" % i)
        with open("conll_input_ep_%d" % i, 'w') as f:
            ep_name = "episode%d" % i
            ids = set([x["id"] for x in user_annots[ep_name]])
            for id in ids:
                user_ex = user_map[ep_name][id]
                test_ex = test_map[ep_name][id]
                assert user_ex["sentence"] == test_ex["sentence"]
                u_sent = user_ex["sentence"].strip().split()
                test_tags = test_ex["tag_sequence"].strip().split()
                user_tags = user_ex["tag_sequence"].strip().split()
                assert len(test_tags) == len(user_tags)

                for orig, gt, pred in zip(u_sent, test_tags, user_tags):
                    f.write("%s %s %s\n" % (orig, gt, pred))
                f.write("\n")

        os.system('%s/evaluation_script/conlleval.pl < conll_input_ep_%d > conll_output_ep_%d' % (os.getcwd(), i, i))

        with open("conll_input_ep_%d" % i) as f:
            print(f.readlines())

        # Read the conll output into an Fscore for this episode.
        def get_fscore(fname):
            with open(fname, 'r') as f:
                for l in f:
                    print(l)
                    l = l.strip()
                    if l.startswith("accuracy"):
                        fscore = float(re.search("FB1:\s*(\d+\.\d+)", l).group(1))
                        return fscore
            return 0.0

        fscores.append(get_fscore("conll_output_ep_%d" % i))

    output = {}
    print("Evaluating for Test Phase")
    output["result"] = [
        {
            "test_split": {
                "Ep. 1": fscores[0],
                "Ep. 2": fscores[1],
                "Ep. 3": fscores[2],
                "Ep. 4": fscores[3],
                "Ep. 5": fscores[4],
                "Average": np.mean(fscores)
            }
        },
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]
    print("Completed evaluation for Test Phase")
    return output
