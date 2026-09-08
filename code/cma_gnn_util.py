import os
import os.path as osp
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def compute_fscore_recover(initial_covered_status, intermediate_covered_status, final_covered_status, info=False):
    cov_uncov = 0
    uncov_cov = 0
    uncov_uncov = 0

    intermediately_covered_num = 0
    intermediately_uncovered_num = 0

    for i in range(len(final_covered_status)):
        is_covered = final_covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        is_intermediately_covered = intermediate_covered_status[i][1]
        is_target = intermediate_covered_status[i][0]

        if is_target != -1:                             # for all points except the head
            if is_intermediately_covered:               # counting up all the points that should stay covered
                intermediately_covered_num += 1
            if not is_intermediately_covered and is_initially_covered:  # counting up points that we want to cover (not including points uncovered from the beginning)
                intermediately_uncovered_num += 1

            if not is_intermediately_covered and is_initially_covered: # if the point was uncovered (compared to the inital state)
                if is_covered:                                         # we want to reward when it is covered back up
                    uncov_cov += 1
                elif not is_covered:                                   # keep count of how many of these points we fail to recover
                    uncov_uncov += 1


            # ! is double counting an issue at all here (since initial state is not being considered)?
            if is_intermediately_covered and not is_covered:  # if this point was covered in the intermediate state, penalize if it got uncovered
                cov_uncov += 1

    total = intermediately_covered_num + intermediately_uncovered_num
    weight = (
        intermediately_uncovered_num / total
        if total > 0
        else 0.0
    )
    penalties = []

    for i in range(1,cov_uncov+1):  # cov_uncov (increasing weight of the PENALTY)
        penalty = i*weight
        penalties.append(penalty if penalty <= 1 else 1)  # maybe want to adjust the weight progression?

    tp = uncov_cov
    fp = np.sum(penalties)
    fn = uncov_uncov
    denominator = tp + 0.5 * (fp + fn)
    f_score = 0.0 if denominator <= 0 else tp / denominator

    if info:
        return f_score, (tp, cov_uncov, fn)

    return f_score

def compute_fscore_uncover(initial_covered_status, final_covered_status):
    targ_uncov = 0
    nontarg_uncov = 0
    targ_cov = 0
    total_nontarg = 0
    nontarg_initially_covered = 0
    nontarg_cov = 0

    for i in range(len(final_covered_status)):
        bod_point_type = final_covered_status[i][0]
        is_covered =  final_covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if bod_point_type == 1:
            if is_covered:
                targ_cov += 1
            else:
                targ_uncov +=1
        elif bod_point_type == 0 and is_initially_covered:
            total_nontarg += 1
            if not is_covered:
                nontarg_uncov += 1
            else:
                nontarg_initially_covered += 1
    # print(total_nontarg)
    total_targ = targ_cov+targ_uncov
    total = total_targ + total_nontarg
    weight = total_targ / total if total > 0 else 0.0
    penalties = []
    for i in range(1,nontarg_uncov+1):
        penalty = i*weight
        penalties.append(penalty if penalty <= 1 else 1)
        # penalty = i*weight if i+weight <= 1 else 1

    # print(penalties)

    tp = targ_uncov
    fp = np.sum(penalties)
    fn = targ_cov
    denominator = tp + 0.5 * (fp + fn)
    f_score = 0.0 if denominator <= 0 else tp / denominator

    # print(targ_uncov + targ_cov, total_nontarg)

    return f_score

def set_x0_for_cmaes(target_limb_code):
    # these actions are unscaled!
    if target_limb_code in [0, 1, 2]:       # Top Right
        x0 = [0.5, -0.4, 0, 0]
        # x0 = [0.5, -0.4, -0.5, -0.5]
    elif target_limb_code in [3, 4, 5]:     # Bottom Right
        x0 = [0.5, 0.5, 0, 0]
    elif target_limb_code in [6, 7, 8]:     # Top Left
        x0 = [-0.5, -0.4, 0, 0]
        # x0 = [-0.5, -0.4, 0.5, -0.5]
    elif target_limb_code in [9, 10, 11]:   # Bottom Left
        x0 = [-0.5, 0.5, 0, 0]
    elif target_limb_code in [13, 15]:
        x0 = [0, 0, 0,  0.5]
    elif target_limb_code in [12, 14]:
        x0 = [0, 0, 0, -0.5]
    else:
        x0 = [0, 0, 0, -0.5]
    return x0


def assemble_fixed_pick_action(pick_fixed, place_xy):
    """Assemble 4D policy action from a locked pick and a (possibly CMA) place."""
    pick = np.asarray(pick_fixed, dtype=np.float64).reshape(-1)[:2]
    place = np.asarray(place_xy, dtype=np.float64).reshape(-1)[:2]
    return np.concatenate([pick, place]).astype(np.float64)


def save_data_to_pickle(idx, seed, recovering, uncover_action, recover_action, human_pose, target_limb_code, sim_info, cma_info, iter_data_dir):
    #! when lines below are uncommented, will not save if no grasp on cloth found
    # if isinstance(covered_status, int) and covered_status == -1:
    #     return

    pid = os.getpid()
    filename = f"tl{target_limb_code}_c{idx}_{seed}_pid{pid}"

    raw_dir = osp.join(iter_data_dir, 'raw')
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    pkl_loc = raw_dir

    final_path = os.path.join(pkl_loc, filename + ".pkl")
    # Write atomically so a worker killed by the watchdog cannot leave a
    # partially written file that later gets mistaken for a completed case.
    temp_path = os.path.join(pkl_loc, f".{filename}.tmp")
    with open(temp_path, "wb") as f:
        pickle.dump({
            "recovering" : recovering,
            "uncover_action":uncover_action,
            "recover_action" : recover_action,
            "human_pose":human_pose,
            'target_limb_code':target_limb_code,
            'sim_info':sim_info,
            'cma_info':cma_info,
            'observation':[sim_info['observation']],
            'info':sim_info['info']}, f)
    os.replace(temp_path, final_path)

def save_dataset(idx, graph, data, sim_info, action, human_pose, covered_status):
    # ! function behavior is not correct at the moment
    if isinstance(covered_status, int) and covered_status == -1:
        return


    initial_blanket_state = sim_info['info']["cloth_initial_subsample"]
    final_blanket_state = sim_info['info']["cloth_final_subsample"]
    cloth_initial, cloth_final = graph.get_cloth_as_tensor(initial_blanket_state, final_blanket_state)

    data['cloth_initial'] = cloth_initial
    data['cloth_final'] = cloth_final
    data['action'] = torch.tensor(action, dtype=torch.float)
    data['human_pose'] = torch.tensor(human_pose, dtype=torch.float)

    proc_data_dir = graph.proc_data_dir
    data = graph.dict_to_Data(data)
    torch.save(data, osp.join(proc_data_dir, f'data_{idx}.pt'))

target_names = [
    '','','Right Arm',
    '','Right Lower Leg','Right Leg',
    '','','Left Arm',
    '','Left Lower Leg','Left Leg',
    'Both Lower Legs','Upper Body','Lower Body',
    'Whole Body'
]
def get_body_point_colors_uncovering(initial_covered_status, covered_status):
    point_colors = []
    for i in range(len(covered_status)):
        is_target = covered_status[i][0]
        is_covered = covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if is_target == 1:
            # infill = 'rgba(168, 102, 39, 1)' if is_covered else 'forestgreen'
            infill = 'rgba(184, 33, 166, 1)' if is_covered else 'forestgreen'
        elif is_target == -1: # head points
            # infill = 'red' if is_covered and not is_initially_covered else 'rgba(255,186,71,1)'
            infill = 'rgba(255, 186, 71, 1)' if not is_covered or is_initially_covered else 'red'
        else:
            infill = 'rgba(255, 186, 71, 1)' if is_covered or not is_initially_covered else 'red'
        point_colors.append(infill)

    return point_colors

def get_body_point_colors_recovering(initial_covered_status, intermediate_covered_status, covered_status):
    point_colors = []
    infill = 0
    for i in range(len(covered_status)):
        is_target = covered_status[i][0]
        is_covered = covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        is_intermediately_covered = intermediate_covered_status[i][1]


        if is_target != -1:
            # Reward for covering points
            if not is_intermediately_covered and is_initially_covered:
                if is_covered:
                    infill = 'rgba(12, 216, 112, 0.8)' #green
                elif not is_covered:
                    infill = 'rgba(215, 199, 239, 0.8)' #purple
            # Penalize for uncovering points
            elif is_intermediately_covered and not is_covered:
                infill = 'rgba(216, 12, 12, 0.8)' #red
            else:
                infill = 'rgba(255, 186, 71, 1)' #tan
        # Penalize for covering the head
        elif is_target == -1 and is_covered and not is_initially_covered:
            infill = 'rgba(236, 165, 165, 0.8)' #pink
        else:
            infill = 'rgba(255, 186, 71, 1)' #tan
        point_colors.append(infill)

    return point_colors

# High-contrast colors for the realized 3-vertex grasp triangle.
GT_ANCHOR_COLORS = (
    'rgba(255, 120, 0, 1)',    # orange
    'rgba(0, 210, 90, 1)',     # green
    'rgba(40, 140, 255, 1)',   # blue
)


def add_gt_anchor_markers(
    fig,
    cloth_points,
    anchor_idx,
    row=1,
    col=2,
    name='GT anchor',
    marker_color=None,
    size=18,
    showlegend=True,
    per_vertex_colors=True,
):
    """Overlay realized grasp triangle vertices on a gen_images-style subplot.

    Default: each of the 3 anchors gets a distinct filled color (not red X).
    """
    cloth = np.asarray(cloth_points, dtype=np.float64)
    if cloth.ndim != 2 or cloth.shape[0] == 0:
        return fig
    anchors = []
    for idx in list(anchor_idx or []):
        i = int(idx)
        if 0 <= i < len(cloth):
            anchors.append(i)
    if not anchors:
        return fig

    if per_vertex_colors:
        for k, ai in enumerate(anchors):
            color = GT_ANCHOR_COLORS[k % len(GT_ANCHOR_COLORS)]
            fig.add_trace(
                go.Scatter(
                    mode='markers',
                    x=[float(cloth[ai, 0])],
                    y=[float(cloth[ai, 1])],
                    name='%s[%d]' % (name, k) if showlegend else name,
                    showlegend=bool(showlegend),
                    marker=dict(
                        color=color,
                        size=size,
                        symbol='circle',
                        line=dict(width=2, color='rgba(0,0,0,1)'),
                    ),
                ),
                row=row,
                col=col,
            )
        return fig

    color = marker_color or GT_ANCHOR_COLORS[0]
    pts = cloth[anchors]
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=pts[:, 0],
            y=pts[:, 1],
            name='%s (n=%d)' % (name, len(anchors)),
            showlegend=bool(showlegend),
            marker=dict(
                color=color,
                size=size,
                symbol='circle',
                line=dict(width=2, color='rgba(0,0,0,1)'),
            ),
        ),
        row=row,
        col=col,
    )
    return fig


def add_action_arrow(fig, action_xy4, col=1, color='rgb(220, 40, 40)', width=4, pick_size=14, pick_name=None, showlegend=False):
    """Add pick marker + place arrow on subplot ``col`` (gen_images axis indexing)."""
    action = np.asarray(action_xy4, dtype=np.float64).reshape(-1)
    if action.size < 4:
        return fig
    xref = 'x%d' % int(col)
    yref = 'y%d' % int(col)
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[float(action[0])],
            y=[float(action[1])],
            name=pick_name or 'grasp',
            showlegend=bool(showlegend),
            marker=dict(color=color, size=pick_size, symbol='diamond',
                        line=dict(width=1.5, color='rgba(0,0,0,1)')),
        ),
        row=1,
        col=int(col),
    )
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.append(
        go.layout.Annotation(
            dict(
                ax=float(action[0]),
                ay=float(action[1]),
                xref=xref,
                yref=yref,
                text='',
                showarrow=True,
                axref=xref,
                ayref=yref,
                x=float(action[2]),
                y=float(action[3]),
                arrowhead=3,
                arrowwidth=width,
                arrowcolor=color,
            )
        )
    )
    fig.update_layout(annotations=annotations)
    return fig


def add_corner_annotation(fig, text, x=1.0, y=1.0, xanchor='right', yanchor='top', font_size=14):
    """Add a gen_images-style corner text box to an existing figure."""
    if not text:
        return fig
    annotation = dict(
        text=str(text),
        x=x,
        y=y,
        xref='paper',
        yref='paper',
        xanchor=xanchor,
        yanchor=yanchor,
        showarrow=False,
        align='right' if xanchor == 'right' else 'left',
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor='rgba(0,0,0,0.25)',
        borderwidth=1,
        font=dict(size=font_size, color='rgba(0,0,0,1)'),
    )
    annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    annotations.append(annotation)
    fig.update_layout(annotations=annotations)
    return fig


def write_gen_images_figure(fig, png_path, html_path=None):
    """Write plotly figure like gen_images.py (PNG preferred, HTML fallback)."""
    png_path = str(png_path)
    html_path = str(html_path) if html_path else (os.path.splitext(png_path)[0] + '.html')
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(png_path)
        return png_path, ''
    except Exception as exc:
        fig.write_html(html_path)
        return html_path, '%s: %s' % (type(exc).__name__, exc)


def generate_figure_data_collection_with_anchors(
    tl,
    uncover_action,
    recover_action,
    body_info,
    all_body_points,
    cloth_initial,
    cloth_intermediate,
    final_cloth,
    recover_anchor_idx=None,
    uncover_anchor_idx=None,
    metrics=None,
    annotation_text=None,
    mark_recover_on_uncover_state=True,
):
    """gen_images data-collection figure + recover grasp/anchors on uncover state.

    Recover grasp happens on ``cloth_intermediate`` (post-uncover). We therefore:
      - draw recover pick/place on the Uncover panel (col=1) where intermediate is shown;
      - draw the same colored 3-anchor highlights on both Uncover (intermediate) and
        Recover (intermediate) panels for correspondence;
      - keep recover action on the Recover panel as well.
    """
    fig = generate_figure_data_collection(
        tl,
        uncover_action,
        recover_action,
        body_info,
        all_body_points,
        cloth_initial,
        cloth_intermediate,
        final_cloth,
        metrics=metrics,
    )

    # Recover grasp/action on uncover (intermediate) state — distinct from black uncover arrow.
    if mark_recover_on_uncover_state and recover_action is not None:
        fig = add_action_arrow(
            fig,
            recover_action,
            col=1,
            color='rgb(220, 40, 40)',
            width=4,
            pick_size=15,
            pick_name='recover grasp (on uncover state)',
            showlegend=True,
        )

    if uncover_anchor_idx:
        fig = add_gt_anchor_markers(
            fig,
            cloth_initial,
            uncover_anchor_idx,
            row=1,
            col=1,
            name='Uncover GT',
            showlegend=True,
            per_vertex_colors=True,
        )

    if recover_anchor_idx:
        # Same colors on both panels (uncover-state / recover-start = intermediate).
        fig = add_gt_anchor_markers(
            fig,
            cloth_intermediate,
            recover_anchor_idx,
            row=1,
            col=1,
            name='Recover GT',
            showlegend=True,
            per_vertex_colors=True,
            size=18,
        )
        fig = add_gt_anchor_markers(
            fig,
            cloth_intermediate,
            recover_anchor_idx,
            row=1,
            col=2,
            name='Recover GT',
            showlegend=False,
            per_vertex_colors=True,
            size=18,
        )

    if annotation_text:
        fig = add_corner_annotation(fig, annotation_text)
    return fig


def generate_figure_data_collection(tl, uncover_action, recover_action, body_info, all_body_points, cloth_initial, cloth_intermediate, final_cloth, metrics=None):
    scale = 4
    num_subplots = 2

    fig = make_subplots(rows=1, cols=num_subplots)
    arrows = []
    bg_color = 'rgba(255,255,255,1)'

    for i in range(num_subplots):
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=all_body_points[:,0],
                y=all_body_points[:,1],
                marker=dict(color='rgba(12, 216, 112, 0.8)', size=10),
                showlegend=False,
            ),
            row=1,
            col=i+1,
        )

    # Uncover panel: initial -> intermediate
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=cloth_initial[:,0],
            y=cloth_initial[:,1],
            showlegend=False,
            marker=dict(color='rgba(99, 190, 242, 0.15)', size=9),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=cloth_intermediate[:,0],
            y=cloth_intermediate[:,1],
            showlegend=False,
            marker=dict(color='rgba(38, 60, 201, 0.55)', size=9),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[uncover_action[0]],
            y=[uncover_action[1]],
            showlegend=False,
            marker=dict(color='rgba(0,0,0,1)', size=12),
        ),
        row=1,
        col=1,
    )
    arrows.append(
        go.layout.Annotation(
            dict(
                ax=uncover_action[0],
                ay=uncover_action[1],
                xref='x1',
                yref='y1',
                text='',
                showarrow=True,
                axref='x1',
                ayref='y1',
                x=uncover_action[2],
                y=uncover_action[3],
                arrowhead=3,
                arrowwidth=4,
                arrowcolor='rgb(0,0,0)',
            )
        )
    )

    # Recover panel: intermediate -> final
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=cloth_intermediate[:,0],
            y=cloth_intermediate[:,1],
            showlegend=False,
            marker=dict(color='rgba(99, 190, 242, 0.35)', size=9),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=final_cloth[:,0],
            y=final_cloth[:,1],
            showlegend=False,
            marker=dict(color='rgba(38, 60, 201, 0.7)', size=9),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[recover_action[0]],
            y=[recover_action[1]],
            showlegend=False,
            marker=dict(color='rgba(0,0,0,1)', size=12),
        ),
        row=1,
        col=2,
    )
    arrows.append(
        go.layout.Annotation(
            dict(
                ax=recover_action[0],
                ay=recover_action[1],
                xref='x2',
                yref='y2',
                text='',
                showarrow=True,
                axref='x2',
                ayref='y2',
                x=recover_action[2],
                y=recover_action[3],
                arrowhead=3,
                arrowwidth=4,
                arrowcolor='rgb(0,0,0)',
            )
        )
    )

    annotations = [
        go.layout.Annotation(
            dict(
                x=0.22,
                y=1.08,
                xref='paper',
                yref='paper',
                text='Uncover',
                showarrow=False,
                font=dict(size=18),
            )
        ),
        go.layout.Annotation(
            dict(
                x=0.78,
                y=1.08,
                xref='paper',
                yref='paper',
                text='Recover',
                showarrow=False,
                font=dict(size=18),
            )
        ),
    ]

    if metrics:
        uncover_reward = metrics.get('uncover_reward')
        uncover_f1 = metrics.get('uncover_f1')
        recover_reward = metrics.get('recover_reward')
        recover_f1 = metrics.get('recover_f1')
        metrics_text = (
            f"Uncover reward={uncover_reward:.2f}, F1={uncover_f1:.3f}"
            f"<br>Recover reward={recover_reward:.2f}, F1={recover_f1:.3f}"
        )
        annotations.append(
            go.layout.Annotation(
                dict(
                    x=0.5,
                    y=-0.05,
                    xref='paper',
                    yref='paper',
                    text=metrics_text,
                    showarrow=False,
                    font=dict(size=16),
                )
            )
        )

    annotations.extend(arrows)

    for i in range(num_subplots):
        fig.update_xaxes(autorange="reversed", visible=False, row=1, col=i+1)
        fig.update_yaxes(autorange="reversed", visible=False, row=1, col=i+1)

    fig.update_layout(
        width=140*scale*2,
        height=195*scale,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        annotations=annotations,
        title={'text': f"Target: {target_names[tl]}", 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
    )

    return fig

def generate_figure_uncover_training(tl, uncover_action, body_info, all_body_points, cloth_initial, cloth_final, metrics=None):
    """Single-panel uncover training visualization (gen_images style)."""
    scale = 4
    bg_color = 'rgba(255,255,255,1)'

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=all_body_points[:, 0],
            y=all_body_points[:, 1],
            marker=dict(color='rgba(12, 216, 112, 0.8)', size=10),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=cloth_initial[:, 0],
            y=cloth_initial[:, 1],
            showlegend=False,
            marker=dict(color='rgba(99, 190, 242, 0.15)', size=9),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=cloth_final[:, 0],
            y=cloth_final[:, 1],
            showlegend=False,
            marker=dict(color='rgba(38, 60, 201, 0.55)', size=9),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[uncover_action[0]],
            y=[uncover_action[1]],
            showlegend=False,
            marker=dict(color='rgba(0,0,0,1)', size=12),
        ),
        row=1,
        col=1,
    )

    annotations = [
        go.layout.Annotation(
            dict(
                ax=uncover_action[0],
                ay=uncover_action[1],
                xref='x1',
                yref='y1',
                text='',
                showarrow=True,
                axref='x1',
                ayref='y1',
                x=uncover_action[2],
                y=uncover_action[3],
                arrowhead=3,
                arrowwidth=4,
                arrowcolor='rgb(0,0,0)',
            )
        ),
        go.layout.Annotation(
            dict(
                x=0.5,
                y=1.08,
                xref='paper',
                yref='paper',
                text='Uncover (training GT)',
                showarrow=False,
                font=dict(size=18),
            )
        ),
    ]

    title = f"Target: {target_names[tl]}"
    if metrics:
        uncover_reward = metrics.get('uncover_reward')
        uncover_f1 = metrics.get('uncover_f1')
        lower_steps = metrics.get('lower_step_count')
        prs = metrics.get('post_release_steps')
        metrics_bits = []
        if uncover_reward is not None:
            metrics_bits.append(f"reward={uncover_reward:.2f}")
        if uncover_f1 is not None:
            metrics_bits.append(f"F1={uncover_f1:.3f}")
        if lower_steps is not None:
            metrics_bits.append(f"lower_steps={lower_steps}")
        if prs is not None:
            metrics_bits.append(f"PRS={prs}")
        if metrics_bits:
            annotations.append(
                go.layout.Annotation(
                    dict(
                        x=0.5,
                        y=-0.08,
                        xref='paper',
                        yref='paper',
                        text=", ".join(metrics_bits),
                        showarrow=False,
                        font=dict(size=16),
                    )
                )
            )

    fig.update_xaxes(autorange='reversed', visible=False, row=1, col=1)
    fig.update_yaxes(autorange='reversed', visible=False, row=1, col=1)
    fig.update_layout(
        width=140 * scale,
        height=195 * scale,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        annotations=annotations,
        title={'text': title, 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
    )
    return fig

def generate_figure_recover(sim_info_fscore, cma_info_fscore, sim_reward, sim_reward_info, cma_reward, cma_reward_info, tl, uncover_action, recover_action, body_info, all_body_points, cloth_initial, final_cloths, cloth_intermediate, initial_covered_status, covered_statuses, fscores, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):
    point_colors_sim = get_body_point_colors_recovering(initial_covered_status[0], initial_covered_status[1], covered_statuses[0])
    point_colors_cma = get_body_point_colors_recovering(initial_covered_status[0], initial_covered_status[1], covered_statuses[1])
    point_colors = [point_colors_sim, point_colors_cma]

    stp, sfp, sfn = sim_info_fscore
    ctp, cfp, cfn = cma_info_fscore

    sim_covered_reward, sim_uncovered_penalty, sim_head_covered_penalty = sim_reward_info
    cma_covered_reward, cma_uncovered_penalty, cma_head_covered_penalty = cma_reward_info

    scale = 4
    num_rows = 1
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

    fig = make_subplots(rows=num_rows, cols=4)
    annotations = []

    fig.add_trace(
        go.Scatter(mode='markers',
                x = cloth_initial[:,0],
                y = cloth_initial[:,1],
                showlegend = False,
                marker=dict(color = 'rgba(99, 190, 242, 0.1)', size = 9)), row=1, col=1)


    for i in range(1,3):
        fig.add_trace(
            go.Scatter(mode='markers',
                        x = all_body_points[:,0],
                        y = all_body_points[:,1],
                        marker=dict(color = 'rgba(255, 186, 71, 1)', size = 10),
                        showlegend=False), row=1, col=i)

    for i, j in enumerate(range(3, 5)):
        fig.add_trace(
            go.Scatter(mode='markers',
                        x = all_body_points[:,0],
                        y = all_body_points[:,1],
                        marker=dict(color = point_colors[i], size = 10),
                        showlegend=False), row=1, col=j)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = cloth_intermediate[:,0],
                y = cloth_intermediate[:,1],
                showlegend = False,
                marker=dict(color = 'rgba(99, 190, 242, 0.5)', size = 9)), row=1, col=2)

    for i in range(3, 5):
        fig.add_trace(
        go.Scatter(mode='markers',
                x = cloth_intermediate[:,0],
                y = cloth_intermediate[:,1],
                showlegend = False,
                marker=dict(color = 'rgba(99, 190, 242, 0.7)', size = 9)), row=1, col=i)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = final_cloths[0][:,0],
                y = final_cloths[0][:,1],
                showlegend = False,
                marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=3)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = final_cloths[1][:,0],
                y = final_cloths[1][:,1],
                showlegend = False,
                marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=4)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = [uncover_action[0]], y = [uncover_action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                row=1, col=2)

    uncover_action_arrow = go.layout.Annotation(dict(
                    ax=uncover_action[0],
                    ay=uncover_action[1],
                    xref=f"x{2}", yref=f"y{2}",
                    text="",
                    showarrow=True,
                    axref=f"x{2}", ayref=f"y{2}",
                    x=uncover_action[2],
                    y=uncover_action[3],
                    arrowhead=3,
                    arrowwidth=4,
                    arrowcolor='rgb(0,0,0)'))
    annotations.append(uncover_action_arrow)


    for i in range(3, 5):
        fig.add_trace(
            go.Scatter(mode='markers',
                    x = [recover_action[0]], y = [recover_action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                    row=1, col=i)

        recover_action_arrow = go.layout.Annotation(dict(
                        ax=recover_action[0],
                        ay=recover_action[1],
                        xref=f"x{i}", yref=f"y{i}",
                        text="",
                        showarrow=True,
                        axref=f"x{i}", ayref=f"y{i}",
                        x=recover_action[2],
                        y=recover_action[3],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))

        annotations.append(recover_action_arrow)

    fscore_text = go.layout.Annotation(dict(x=-0.6, y=1.2, xref='x', yref='y', text=f'sim true pos: {stp} sim false pos: {sfp} sim false neg: {sfn}<br>cma true pos: {ctp}\n cma false pos: {cfp}\n cma false neg: {cfn}', showarrow=False),  font=dict(size=16))
    sim_reward_text = go.layout.Annotation(dict(x=-0.6, y=-1.2, xref='x', yref='y', text=f'Sim Reward: {int(sim_reward)}, Uncov Penalty: {int(sim_uncovered_penalty)}, Cov Reward: {int(sim_covered_reward)}, Head Cov Penalty: {int(sim_head_covered_penalty)}', showarrow=False), font=dict(size=16))
    cma_reward_text = go.layout.Annotation(dict(x=-0.6, y=-1, xref='x', yref='y', text=f'CMA Reward: {int(cma_reward)}, Uncov Penalty: {int(cma_uncovered_penalty)}, Cov Reward: {int(cma_covered_reward)}, Head Cov Penalty: {int(cma_head_covered_penalty)}', showarrow=False), font=dict(size=16))

    annotations.append(fscore_text)
    annotations.append(sim_reward_text)
    annotations.append(cma_reward_text)

    for i in range(1,5):
        fig.update_xaxes(autorange="reversed", visible=False, row=1, col=i)
        fig.update_yaxes(autorange="reversed", visible=False, row=1, col=i)

    fig.update_layout(width=100*scale*4, height=200*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=annotations,
                        title={'text': f"Target: {target_names[tl]}<br>Sim F-Score = {fscores[0]:.2f}<br>CMA F-Score = {fscores[1]:.2f}",'y':0.08,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

    return fig
def generate_figure_uncover(sim_reward, cma_reward, tl, uncover_action, body_info, all_body_points, cloth_initial, final_cloths, initial_covered_status, covered_statuses, fscores, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):
    point_colors_sim = get_body_point_colors_uncovering(initial_covered_status[0], covered_statuses[0])
    point_colors_cma = get_body_point_colors_uncovering(initial_covered_status[0], covered_statuses[1])
    point_colors = [point_colors_sim, point_colors_cma]

    scale = 4
    num_rows = 1
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

    fig = make_subplots(rows=num_rows, cols=4)
    annotations = []

    for i in range(1, 4):
        fig.add_trace(
            go.Scatter(mode='markers',
                    x = cloth_initial[:,0],
                    y = cloth_initial[:,1],
                    showlegend = False,
                    marker=dict(color = 'rgba(99, 190, 242, 0.1)', size = 9)), row=1, col=i)


    fig.add_trace(
        go.Scatter(mode='markers',
                    x = all_body_points[:,0],
                    y = all_body_points[:,1],
                    marker=dict(color = 'rgba(255, 186, 71, 1)', size = 10),
                    showlegend=False), row=1, col=1)

    for i, j in enumerate(range(2, 4)):
        fig.add_trace(
            go.Scatter(mode='markers',
                        x = all_body_points[:,0],
                        y = all_body_points[:,1],
                        marker=dict(color = point_colors[i], size = 10),
                        showlegend=False), row=1, col=j)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = final_cloths[0][:,0],
                y = final_cloths[0][:,1],
                showlegend = False,
                marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=2)

    fig.add_trace(
        go.Scatter(mode='markers',
                x = final_cloths[1][:,0],
                y = final_cloths[1][:,1],
                showlegend = False,
                marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=3)

    for i in range(2, 4):
        fig.add_trace(
            go.Scatter(mode='markers',
                    x = [uncover_action[0]], y = [uncover_action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                    row=1, col=i)

        action_arrow = go.layout.Annotation(dict(
                        ax=uncover_action[0],
                        ay=uncover_action[1],
                        xref=f"x{i}", yref=f"y{i}",
                        text="",
                        showarrow=True,
                        axref=f"x{i}", ayref=f"y{i}",
                        x=uncover_action[2],
                        y=uncover_action[3],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))


        annotations.append(action_arrow)

    for i in range(1,4):
        fig.update_xaxes(autorange="reversed", visible=False, row=1, col=i)
        fig.update_yaxes(autorange="reversed", visible=False, row=1, col=i)

    fig.update_layout(width=100*scale*4, height=200*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=annotations,
                        title={'text': f"Target: {target_names[tl]}<br>Sim F-Score = {fscores[0]:.2f}<br>CMA F-Score = {fscores[1]:.2f}",'y':0.08,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

    return fig

def generate_figure_uncover_open_loop(
    tl,
    uncover_action,
    all_body_points,
    cloth_initial,
    gt_cloth,
    pred_cloth,
    initial_covered_status,
    gt_covered_status,
    pred_covered_status,
    fscores,
    metrics=None,
    include_overlay=True,
    transparent=False,
):
    """Open-loop validation figure: GT vs model-predicted uncover cloth."""
    point_colors_gt = get_body_point_colors_uncovering(initial_covered_status, gt_covered_status)
    point_colors_pred = get_body_point_colors_uncovering(initial_covered_status, pred_covered_status)

    scale = 4
    num_cols = 5 if include_overlay else 4
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

    fig = make_subplots(rows=1, cols=num_cols)
    annotations = []

    for col in range(1, num_cols + 1):
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=cloth_initial[:, 0],
                y=cloth_initial[:, 1],
                showlegend=False,
                marker=dict(color='rgba(99, 190, 242, 0.1)', size=9),
            ),
            row=1,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=all_body_points[:, 0],
            y=all_body_points[:, 1],
            marker=dict(color='rgba(255, 186, 71, 1)', size=10),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for col, colors in ((2, point_colors_gt), (3, point_colors_pred)):
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=all_body_points[:, 0],
                y=all_body_points[:, 1],
                marker=dict(color=colors, size=10),
                showlegend=False,
            ),
            row=1,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=gt_cloth[:, 0],
            y=gt_cloth[:, 1],
            showlegend=False,
            marker=dict(color='rgba(38, 60, 201, 0.55)', size=9),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=pred_cloth[:, 0],
            y=pred_cloth[:, 1],
            showlegend=False,
            marker=dict(color='rgba(38, 60, 201, 0.55)', size=9),
        ),
        row=1,
        col=3,
    )

    if include_overlay:
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=gt_cloth[:, 0],
                y=gt_cloth[:, 1],
                showlegend=False,
                marker=dict(color='rgba(99, 190, 242, 0.35)', size=8),
            ),
            row=1,
            col=4,
        )
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=pred_cloth[:, 0],
                y=pred_cloth[:, 1],
                showlegend=False,
                marker=dict(color='rgba(220, 53, 69, 0.45)', size=8),
            ),
            row=1,
            col=4,
        )
        annotations.append(
            go.layout.Annotation(
                dict(
                    x=0.72,
                    y=1.08,
                    xref='paper',
                    yref='paper',
                    text='Overlay (GT blue / Pred red)',
                    showarrow=False,
                    font=dict(size=14),
                )
            )
        )

    for col in (2, 3):
        fig.add_trace(
            go.Scatter(
                mode='markers',
                x=[uncover_action[0]],
                y=[uncover_action[1]],
                showlegend=False,
                marker=dict(color='rgba(0,0,0,1)', size=12),
            ),
            row=1,
            col=col,
        )
        annotations.append(
            go.layout.Annotation(
                dict(
                    ax=uncover_action[0],
                    ay=uncover_action[1],
                    xref=f'x{col}',
                    yref=f'y{col}',
                    text='',
                    showarrow=True,
                    axref=f'x{col}',
                    ayref=f'y{col}',
                    x=uncover_action[2],
                    y=uncover_action[3],
                    arrowhead=3,
                    arrowwidth=4,
                    arrowcolor='rgb(0,0,0)',
                )
            )
        )

    col_labels = ['Initial', 'GT', 'Pred']
    if include_overlay:
        col_labels.append('Overlay')
    for idx, label in enumerate(col_labels):
        annotations.append(
            go.layout.Annotation(
                dict(
                    x=(idx + 0.5) / num_cols,
                    y=1.08,
                    xref='paper',
                    yref='paper',
                    text=label,
                    showarrow=False,
                    font=dict(size=16),
                )
            )
        )

    if metrics:
        metrics_text = (
            f"RMSE={metrics.get('vertex_rmse', float('nan')):.4f}, "
            f"GT F1={metrics.get('gt_f1', float('nan')):.3f}, "
            f"Pred F1={metrics.get('pred_f1', float('nan')):.3f}, "
            f"reward gap={metrics.get('reward_gap', float('nan')):.2f}"
        )
        annotations.append(
            go.layout.Annotation(
                dict(
                    x=0.5,
                    y=-0.08,
                    xref='paper',
                    yref='paper',
                    text=metrics_text,
                    showarrow=False,
                    font=dict(size=14),
                )
            )
        )

    for col in range(1, num_cols + 1):
        fig.update_xaxes(autorange='reversed', visible=False, row=1, col=col)
        fig.update_yaxes(autorange='reversed', visible=False, row=1, col=col)

    fig.update_layout(
        width=100 * scale * num_cols,
        height=200 * scale,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        annotations=annotations,
        title={
            'text': (
                f"Target: {target_names[tl]}<br>"
                f"GT F-Score = {fscores[0]:.2f}<br>"
                f"Pred F-Score = {fscores[1]:.2f}"
            ),
            'y': 0.08,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom',
        },
    )
    return fig

def generate_comparison(tl, action, body_info, all_body_points, cloth_initial, cloth_intermediate, cloth_final, pred, initial_covered_status, covered_statuses, fscores, plot_initial=False, transparent=False, draw_axes =False):
    scale = 4
    num_subplots = 2
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

    fig = make_subplots(rows=1, cols=num_subplots)
    arrows = []
    for i in range(num_subplots):

        point_colors = get_body_point_colors_recovering(initial_covered_status, covered_statuses[i])

        if plot_initial:
            fig.add_trace(
                go.Scatter(mode='markers',
                        x = cloth_initial[:,0],
                        y = cloth_initial[:,1],
                        showlegend = False,
                        marker=dict(color = 'rgba(99, 190, 242, 0.1)', size = 9)), row=1, col=i+1)
        fig.add_trace(
            go.Scatter(mode='markers',
                        x = all_body_points[:,0],
                        y = all_body_points[:,1],
                        marker=dict(color = point_colors, size = 10),
                        showlegend=False), row=1, col=i+1)

        # fig.add_trace(
        #     go.Scatter(mode='markers',
        #             x = cloth_intermediate[:,0],
        #             y = cloth_intermediate[:,1],
        #             showlegend = False,
        #             marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=i+1)

        fig.add_trace(
            go.Scatter(mode='markers',
                    x = cloth_final[:,0],
                    y = cloth_final[:,1],
                    showlegend = False,
                    marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=1)

        fig.add_trace(
            go.Scatter(mode='markers',
                    x = pred[:,0],
                    y = pred[:,1],
                    showlegend = False,
                    marker=dict(color = 'rgba(38, 60, 201, 0.5)', size = 9)), row=1, col=2)

        fig.add_trace(
            go.Scatter(mode='markers',
                    x = [action[0]], y = [action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                    row=1, col=i+1)

        action_arrow = go.layout.Annotation(dict(
                        ax=action[4],
                        ay=action[5],
                        xref=f"x{i+1}", yref=f"y{i+1}",
                        text="",
                        showarrow=True,
                        axref=f"x{i+1}", ayref=f"y{i+1}",
                        x=action[6],
                        y=action[7],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))

        arrows.append(action_arrow)

        fig.update_xaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
        fig.update_yaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)

    fig.update_layout(width=2*140*scale, height=195*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=arrows,
                        title={'text': f"Target: {target_names[tl]}<br>F-Score = {fscores[0]:.2f}",'y':0.08,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

    # fig.show()
    return fig


def generate_figure_recovering(action, all_body_points, cloth_initial, cloth_intermediate, cloth_final, pred, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):
    scale = 4
    num_subplots = 2 if compare_subplots else 1
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

    # fig = go.Figure()
    fig = make_subplots(rows=2, cols=num_subplots)
    arrows = []

    for i in range(num_subplots):

        # print(i)
        #* For bed plotting
        fig.add_shape(type="rect",
            x0=0.44, y0=1.05, x1=-0.44, y1=-1.05,
            line=dict(color='rgb(163, 163, 163)'), fillcolor = 'rgb(163, 163, 163)', opacity=0.2, layer='below', row=1, col=i+1)

        #Light Blue
        if plot_initial:
            fig.add_trace(
                go.Scatter(mode='markers',
                           x = cloth_initial[:,0],
                           y = cloth_initial[:,1],
                           showlegend = False,
                           marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=1)

        # TODO: don't need to change point colors based on covered/uncovered
        # fig.add_trace(
        #     go.Scatter(mode='markers',
        #                x = all_body_points[:,0],
        #                y = all_body_points[:,1],
        #                showlegend=False,
        #                marker=dict(color = 'rgba(240, 30, 200, 1)')), row=1, col=i+1)

        # fig.add_trace(
        #             go.Scatter(mode='markers',
        #                     x = all_body_points[:,0],
        #                     y = all_body_points[:,1],
        #                     showlegend=False,
        #                     marker=dict(color = 'rgba(240, 30, 200, 1)')), row=2, col=i+1)
        #Dark Blue
        fig.add_trace(
            go.Scatter(mode='markers',
                x = cloth_intermediate[:,0],
                y = cloth_intermediate[:,1],
                showlegend = False,
                marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=2)

        fig.add_trace(
            go.Scatter(mode='markers',
                       x = cloth_final[i][:,0],
                       y = cloth_final[i][:,1],
                       showlegend = False,
                       marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=3)

        fig.add_trace(
            go.Scatter(mode='markers',
                       x = cloth_final[:,0],
                       y = cloth_final[:,1],
                       showlegend = False,
                       marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=2, col=1)
        fig.add_trace(
            go.Scatter(mode='markers',
                       x = pred[:,0],
                       y = pred[:,1],
                       showlegend = False,
                       marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=2, col=2)

        fig.add_trace(
            go.Scatter(mode='markers',
                       x = [action[0]], y = [action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                       row=1, col=[1])
        fig.add_trace(
            go.Scatter(mode='markers',
                       x = [action[2]], y = [action[3]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                       row=1, col=[2])

        action_arrow_intermediate = go.layout.Annotation(dict(
                        ax=action[0],
                        ay=action[1],
                        xref=f"x{2}", yref=f"y{2}",
                        text="",
                        showarrow=True,
                        axref=f"x{2}", ayref=f"y{2}",
                        x=action[2],
                        y=action[3],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))
        action_arrow_final = go.layout.Annotation(dict(
                        ax=action[4],
                        ay=action[5],
                        xref=f"x{3}", yref=f"y{3}",
                        text="",
                        showarrow=True,
                        axref=f"x{3}", ayref=f"y{3}",
                        x=action[6],
                        y=action[7],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))
        arrows += [action_arrow_intermediate, action_arrow_final]
        fig.update_xaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
        fig.update_yaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)

    fig.update_layout(width=140*3*scale, height=195*2*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=arrows,
                         title={'text': f"", 'y':0.05,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

    # fig.show()
    return fig

# def generate_figure(action, all_body_points, cloth_initial, cloth_final, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):
#     scale = 4
#     num_subplots = 3 if compare_subplots else 1
#     bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

#     # fig = go.Figure()
#     fig = make_subplots(rows=1, cols=num_subplots)
#     arrows = []

#     for i in range(num_subplots):

#         # print(i)
#         #* For bed plotting
#         fig.add_shape(type="rect",
#             x0=0.44, y0=1.05, x1=-0.44, y1=-1.05,
#             line=dict(color='rgb(163, 163, 163)'), fillcolor = 'rgb(163, 163, 163)', opacity=0.2, layer='below', row=1, col=i+1)

#         #Light Blue
#         if plot_initial:
#             fig.add_trace(
#                 go.Scatter(mode='markers',
#                         x = cloth_initial[:,0],
#                         y = cloth_initial[:,1],
#                         showlegend = False,
#                         marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=1)

#         # TODO: don't need to change point colors based on covered/uncovered
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                     x = all_body_points[:,0],
#                     y = all_body_points[:,1],
#                     showlegend=False,
#                     marker=dict(color = 'rgba(240, 30, 200, 1)')), row=1, col=i+1)

#         fig.add_trace(
#             go.Scatter(mode='markers',
#                 x = cloth_final[:,0],
#                 y = cloth_final[:,1],
#                 showlegend = False,
#                 marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=3)
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                         x = [action[0]], y = [action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
#                         row=1, col=[1])
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                         x = [action[2]], y = [action[3]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
#                         row=1, col=[2])
#         # if plotting more than one action arrow turns out to be annoying don't worry about it
#         action_arrow_initial = go.layout.Annotation(dict(
#                         ax=action[0],
#                         ay=action[1],
#                         xref=f"x{1}", yref=f"y{1}",
#                         text="",
#                         showarrow=True,
#                         axref=f"x{1}", ayref=f"y{1}",
#                         x=action[2],
#                         y=action[3],
#                         arrowhead=3,
#                         arrowwidth=4,
#                         arrowcolor='rgb(0,0,0)'))

#         arrows += [action_arrow_initial]
#         fig.update_xaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
#         fig.update_yaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)

#     fig.update_layout(width=140*3*scale, height=195*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=arrows,
#                         title={'text': f"", 'y':0.05,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

#     # fig.show()
#     return fig



# def generate_figure_3_states(action, all_body_points, cloth_initial, cloth_intermediate, cloth_final, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):
#     scale = 4
#     num_subplots = 3 if compare_subplots else 1
#     bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'

#     # fig = go.Figure()
#     fig = make_subplots(rows=1, cols=num_subplots)
#     arrows = []

#     for i in range(num_subplots):

#         # print(i)
#         #* For bed plotting
#         fig.add_shape(type="rect",
#             x0=0.44, y0=1.05, x1=-0.44, y1=-1.05,
#             line=dict(color='rgb(163, 163, 163)'), fillcolor = 'rgb(163, 163, 163)', opacity=0.2, layer='below', row=1, col=i+1)

#         #Light Blue
#         if plot_initial:
#             fig.add_trace(
#                 go.Scatter(mode='markers',
#                         x = cloth_initial[:,0],
#                         y = cloth_initial[:,1],
#                         showlegend = False,
#                         marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=1)

#         # TODO: don't need to change point colors based on covered/uncovered
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                     x = all_body_points[:,0],
#                     y = all_body_points[:,1],
#                     showlegend=False,
#                     marker=dict(color = 'rgba(240, 30, 200, 1)')), row=1, col=i+1)

#         #Dark Blue
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                 x = cloth_intermediate[:,0],
#                 y = cloth_intermediate[:,1],
#                 showlegend = False,
#                 marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=2)

#         fig.add_trace(
#             go.Scatter(mode='markers',
#                     x = cloth_final[:,0],
#                     y = cloth_final[:,1],
#                     showlegend = False,
#                     marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=3)
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                     x = [action[0]], y = [action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
#                     row=1, col=[1])
#         fig.add_trace(
#             go.Scatter(mode='markers',
#                     x = [action[2]], y = [action[3]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
#                     row=1, col=[2])
#         # if plotting more than one action arrow turns out to be annoying don't worry about it
#         action_arrow_initial = go.layout.Annotation(dict(
#                         ax=action[0],
#                         ay=action[1],
#                         xref=f"x{1}", yref=f"y{1}",
#                         text="",
#                         showarrow=True,
#                         axref=f"x{1}", ayref=f"y{1}",
#                         x=action[2],
#                         y=action[3],
#                         arrowhead=3,
#                         arrowwidth=4,
#                         arrowcolor='rgb(0,0,0)'))
#         action_arrow_intermediate = go.layout.Annotation(dict(
#                         ax=action[2],
#                         ay=action[3],
#                         xref=f"x{2}", yref=f"y{2}",
#                         text="",
#                         showarrow=True,
#                         axref=f"x{2}", ayref=f"y{2}",
#                         x=action[0],
#                         y=action[1],
#                         arrowhead=3,
#                         arrowwidth=4,
#                         arrowcolor='rgb(0,0,0)'))
#         arrows += [action_arrow_initial, action_arrow_intermediate]
#         fig.update_xaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
#         fig.update_yaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)

#     fig.update_layout(width=140*3*scale, height=195*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=arrows,
#                         title={'text': f"", 'y':0.05,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

#     # fig.show()
#     return fig
