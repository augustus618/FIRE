import time

from data_flow_analysis import global_data_flow_analysis
from fbdconfig import debug


def decode(pred_scores, threshold=0.5):
    return {addr: (1 if pred_scores[addr] >= threshold else 0) for addr in pred_scores}


def construct_start_pcs(instruction_sequence, basic_blocks, body_tags, fallback_tag, tag_id_to_pc, pred_scores,
                        threshold=0.5, not_starts=None):
    if not_starts is None:
        not_starts = set()

    init_starts = set()

    for body_tag in body_tags:
        init_starts.add(tag_id_to_pc[body_tag])

    if len(fallback_tag) != 0:
        init_starts.add(tag_id_to_pc[list(fallback_tag)[0]])

    decoded = decode(pred_scores, threshold)

    pred_tag_dict = decoded
    for pc in pred_tag_dict:
        if pred_tag_dict[pc] == 1 and pc not in not_starts:
            init_starts.add(pc)
    return init_starts


def detect_func(instruction_sequence, basic_blocks, pc_to_instruction_index, tag_id_to_pc,
                external_function_entry_tag_to_body_tag, fallback_tag,
                pred_scores, threshold1=0.5, threshold2=0.3, delay=0.05):
    invalid_calls = set()
    body_tags = external_function_entry_tag_to_body_tag.values()
    start = time.time()
    func_starts = construct_start_pcs(instruction_sequence, basic_blocks, body_tags, fallback_tag, tag_id_to_pc,
                                      pred_scores, threshold1)
    removed_time = time.time() - start

    possible_calls = {}
    wait_for_exporation = func_starts
    not_starts = set()
    fbs = {}
    call_graph = {}  # entry -> call_site_index -> (tgt_func, tag_context)

    while len(wait_for_exporation):
        delay_flag = False
        _wait_for_exporation = set()
        for entry_pc in wait_for_exporation:
            fb = set()
            _call_graph = {}
            missing_flag, invalid_flag, _invalid_calls = global_data_flow_analysis(instruction_sequence,
                                                                                   pc_to_instruction_index,
                                                                                   entry_pc, func_starts,
                                                                                   possible_calls, invalid_calls, fb,
                                                                                   _call_graph, threshold1,
                                                                                   threshold2, fbs.keys())
            if (not missing_flag) and (not invalid_flag):
                fbs[entry_pc] = fb
                call_graph[entry_pc] = _call_graph
            else:
                if len(_invalid_calls) > 0 and debug:
                    print("find the invalid calls at {}".format(_invalid_calls))
                _wait_for_exporation.add(entry_pc)
                delay_flag = delay_flag or missing_flag
                invalid_calls.update(_invalid_calls)
                for call_index, target in _invalid_calls.items():
                    if target in possible_calls and call_index in possible_calls[target]:
                        # assert call[1] in possible_calls
                        possible_calls[target].remove(call_index)
                        pass

        # for target, call_site in possible_calls.items():
        #     if len(call_site) == 0 and target in fbs:
        #         not_starts.add(target)
        #         fbs.pop(target)
        #         if debug:
        #             print("function start with " + str(target) + " have been removed")

        _fbs = {}
        new_call_graph = {}
        for start in fbs.keys():
            start_tag = instruction_sequence[pc_to_instruction_index[start]].tag_id
            if start_tag in body_tags or start_tag in fallback_tag:
                _fbs[start] = fbs[start]
                new_call_graph[start] = call_graph[start]
            elif start not in possible_calls or len(possible_calls[start]) == 0:
                not_starts.add(start)
                if debug:
                    print("function start with " + str(start) + " have been removed")
            else:
                _fbs[start] = fbs[start]
                new_call_graph[start] = call_graph[start]
        fbs = _fbs
        call_graph = new_call_graph

        func_starts -= not_starts
        _wait_for_exporation -= not_starts

        if delay_flag and threshold1 >= threshold2:
            if debug:
                print("the threshold1 have been reduce {}, current threshold is {}".format(delay_flag, threshold1))
            threshold1 -= delay
            _func_starts = construct_start_pcs(instruction_sequence, basic_blocks, body_tags, fallback_tag,
                                               tag_id_to_pc, pred_scores, threshold=threshold1,
                                               not_starts=not_starts)
            new_func_starts = set(_func_starts) - set(func_starts)
            if len(new_func_starts) > 0 and debug:
                print("explore more functions {}".format(new_func_starts))
            _wait_for_exporation |= set(_func_starts) - set(func_starts)
            func_starts = _func_starts

        wait_for_exporation = _wait_for_exporation

    return fbs, call_graph, removed_time
