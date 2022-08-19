from typing import List

import torch
import torch.nn.functional as F


class DumbContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def compute_realignment_loss(
    encoder,
    realignment_transformation,
    realignment_layers: List[int],
    strong_alignment=False,
    realignment_temperature=0.1,
    realignment_coef=1.0,
    alignment_hidden_size=128,
    no_backward_for_source=False,
    regularization_lambda=1.0,
    initial_model=None,
    # sentences from left language
    left_input_ids=None,
    left_attention_mask=None,
    left_token_type_ids=None,
    left_position_ids=None,
    left_head_mask=None,
    left_inputs_embeds=None,
    # sentences from right language
    right_input_ids=None,
    right_attention_mask=None,
    right_token_type_ids=None,
    right_position_ids=None,
    right_head_mask=None,
    right_inputs_embeds=None,
    # alignment labels
    alignment_left_ids=None,  # [word_id] -> [[word_id | -1]]
    alignment_left_positions=None,  # [[batch, start, end]] -> [[[start, end]]]
    alignment_right_ids=None,  # [word_id]
    alignment_right_positions=None,  # [[batch, start, end]]
    alignment_nb=None,  # [[i]]
    alignment_left_length=None,  # [[i]]
    alignment_right_length=None,  # [[i]]
):
    total_loss = None

    context_manager = torch.no_grad() if no_backward_for_source else DumbContext()

    with context_manager:
        left_output = encoder(
            left_input_ids,
            attention_mask=left_attention_mask,
            token_type_ids=left_token_type_ids,
            position_ids=left_position_ids,
            head_mask=left_head_mask,
            inputs_embeds=left_inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        left_hidden_states = left_output.hidden_states

    if initial_model is not None:
        initial_output = initial_model(
            left_input_ids,
            attention_mask=left_attention_mask,
            token_type_ids=left_token_type_ids,
            position_ids=left_position_ids,
            head_mask=left_head_mask,
            inputs_embeds=left_inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        initial_hidden_states = initial_output.hidden_states

    right_output = encoder(
        right_input_ids,
        attention_mask=right_attention_mask,
        token_type_ids=right_token_type_ids,
        position_ids=right_position_ids,
        head_mask=right_head_mask,
        inputs_embeds=right_inputs_embeds,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
    )

    right_hidden_states = right_output.hidden_states

    the_device = left_hidden_states[0].device

    for layer in realignment_layers:
        # Inspired by https://github.com/shijie-wu/crosslingual-nlp/blob/780f738df2b75f653aaaf11b9f513850fe11ba36/src/model/aligner.py#L139

        aligned_left_repr = torch.zeros(
            (alignment_left_ids.shape[0], alignment_left_ids.shape[1], alignment_hidden_size),
            device=the_device,
        )
        aligned_right_repr = torch.zeros(
            (alignment_left_ids.shape[0], alignment_left_ids.shape[1], alignment_hidden_size),
            device=the_device,
        )

        for b in range(alignment_left_ids.shape[0]):
            for i in range(alignment_left_ids.shape[1]):
                aligned_left_repr[b, i] = realignment_transformation(
                    torch.sum(
                        left_hidden_states[layer][
                            b,
                            alignment_left_positions[b, alignment_left_ids[b, i]][
                                0
                            ] : alignment_left_positions[b, alignment_left_ids[b, i]][1],
                        ],
                        0,
                    )
                )
                aligned_right_repr[b, i] = realignment_transformation(
                    torch.sum(
                        right_hidden_states[layer][
                            b,
                            alignment_right_positions[b, alignment_right_ids[b, i]][
                                0
                            ] : alignment_right_positions[b, alignment_right_ids[b, i]][1],
                        ],
                        0,
                    )
                )

        all_left_repr = torch.zeros(
            (
                alignment_left_positions.shape[0],
                alignment_left_positions.shape[1],
                alignment_hidden_size,
            ),
            device=the_device,
        )
        all_right_repr = torch.zeros(
            (
                alignment_right_positions.shape[0],
                alignment_right_positions.shape[1],
                alignment_hidden_size,
            ),
            device=the_device,
        )
        for b in range(alignment_right_positions.shape[0]):
            for i in range(alignment_left_positions.shape[1]):
                all_left_repr[b, i] = realignment_transformation(
                    torch.sum(
                        left_hidden_states[layer][
                            b,
                            alignment_left_positions[b, i][0] : alignment_left_positions[b, i][1],
                        ],
                        0,
                    )
                )
            for i in range(alignment_right_positions.shape[1]):
                all_right_repr[b, i] = realignment_transformation(
                    torch.sum(
                        right_hidden_states[layer][
                            b,
                            alignment_right_positions[b, i][0] : alignment_right_positions[b, i][1],
                        ],
                        0,
                    )
                )

        aligned_left_repr = torch.cat(
            (*[aligned_left_repr[b][: alignment_nb[b]] for b in range(aligned_left_repr.shape[0])],)
        )
        aligned_right_repr = torch.cat(
            (
                *[
                    aligned_right_repr[b][: alignment_nb[b]]
                    for b in range(aligned_right_repr.shape[0])
                ],
            )
        )
        all_left_repr = torch.cat(
            (
                *[
                    all_left_repr[b][: alignment_left_length[b]]
                    for b in range(all_left_repr.shape[0])
                ],
            )
        )
        all_right_repr = torch.cat(
            (
                *[
                    all_right_repr[b][: alignment_right_length[b]]
                    for b in range(all_right_repr.shape[0])
                ],
            )
        )

        right_cumul_length = torch.cat(
            (
                torch.tensor([0], dtype=torch.long, device=the_device),
                torch.cumsum(alignment_right_length, 0),
            )
        )
        left_cumul_length = torch.cat(
            (
                torch.tensor([0], dtype=torch.long, device=the_device),
                torch.cumsum(alignment_left_length, 0),
            )
        )

        left_goal = torch.cat(
            (
                *[
                    all_left_repr.shape[0]
                    + right_cumul_length[b]
                    + alignment_right_ids[b][: alignment_nb[b]]
                    for b in range(alignment_left_ids.shape[0])
                ],
            )
        )
        right_goal = torch.cat(
            (
                *[
                    left_cumul_length[b] + alignment_left_ids[b][: alignment_nb[b]]
                    for b in range(alignment_right_ids.shape[0])
                ],
            )
        )

        aligned_reprs = torch.cat((aligned_left_repr, aligned_right_repr))
        all_reprs = torch.cat((all_left_repr, all_right_repr))
        sim = torch.matmul(aligned_reprs, all_reprs.transpose(0, 1))
        aligned_norms = aligned_reprs.norm(dim=1, keepdim=True)
        all_norms = all_reprs.norm(dim=1, keepdim=True)

        sim /= aligned_norms
        sim /= all_norms.transpose(0, 1)
        sim /= realignment_temperature

        if not strong_alignment:
            # remove same-language similarities
            sim[: aligned_left_repr.shape[0], : all_left_repr.shape[0]] -= 1e6
            sim[aligned_left_repr.shape[0] :, all_left_repr.shape[0] :] -= 1e6
        else:
            # remove (x,x) similarities
            sim[
                torch.arange(0, aligned_left_repr.shape[0], 1, device=the_device),
                right_goal,
            ] -= 1e6
            sim[
                torch.arange(
                    aligned_right_repr.shape[0],
                    2 * aligned_right_repr.shape[0],
                    1,
                    device=the_device,
                ),
                left_goal,
            ] -= 1e6

        logits = F.log_softmax(sim, dim=-1)
        goal = torch.cat((left_goal, right_goal))

        loss = F.nll_loss(logits, goal)

        if initial_model is not None:
            loss += regularization_lambda * F.mse_loss(
                left_hidden_states[layer], initial_hidden_states[layer]
            )

        if total_loss is None:
            total_loss = (realignment_coef / len(realignment_layers)) * loss
        else:
            total_loss += (realignment_coef / len(realignment_layers)) * loss

    return total_loss
