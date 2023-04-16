import pandas as pd
from bokeh.models import HoverTool
from bokeh.palettes import all_palettes
from bokeh.plotting import figure
import streamlit as st

from src.evaluation.evaluator import Evaluator
from src.submissions.submissions_manager import SingleParticipantSubmissions


class PersonalProgress:
    def __init__(self, participant_submissions: SingleParticipantSubmissions, evaluator: Evaluator):
        self.participant_submissions = participant_submissions
        self.evaluator = evaluator
        self.metric_names = [metric.name() for metric in self.evaluator.metrics()]
        self.submission_name_column = 'Submission name'
        self.submission_time_column = 'Submission time'

    def show_progress(self, progress_plot_placeholder = None):
        bokeh_plot = self._get_bokeh_progress_plot()
        if progress_plot_placeholder is not None:
            progress_plot_placeholder.bokeh_chart(bokeh_plot, use_container_width=True)
        else:
            st.bokeh_chart(bokeh_plot, use_container_width=True)

    @st.cache(hash_funcs={SingleParticipantSubmissions: lambda x: x.submissions_hash()},
              allow_output_mutation=True, show_spinner=False)
    def _get_bokeh_progress_plot(self):
        self.participant_submissions.update_results(evaluator=self.evaluator)

        progress_df = pd.DataFrame([[self.participant_submissions.get_submission_name_from_path(submission_filepath),
                                     self.participant_submissions.get_datetime_from_path(submission_filepath),
                                     *[res.value for res in submission_results]]
                                    for submission_filepath, submission_results in
                                    self.participant_submissions.results.items()],
                                   columns=[self.submission_name_column, self.submission_time_column,
                                            *self.metric_names])

        progress_df.sort_values(by=self.submission_time_column, inplace=True)

        return self._create_bokeh_plot_from_df(progress_df, self.submission_time_column,
                                               self.submission_name_column)

    def _create_bokeh_plot_from_df(self, progress_df: pd.DataFrame,
                                   submission_time_col: str, submission_name_col: str):
        p = figure(x_axis_type='datetime')
        for column, color in zip(self.metric_names, get_colormap(len(self.metric_names))):
            glyph_line = p.line(
                x=submission_time_col,
                y=column,
                legend_label=" " + column,
                source=progress_df,
                color=color,
            )
            glyph_scatter = p.scatter(
                x=submission_time_col,
                y=column,
                legend_label=" " + column,
                source=progress_df,
                color=color,
                marker='circle',
                fill_alpha=0.3,
            )
            p.add_tools(HoverTool(
                tooltips=[(submission_time_col, f'@{{{submission_time_col}}}{{%Y-%m-%d %H:%M}}'),
                          (submission_name_col, f'@{{{submission_name_col}}}'),
                          (column, f'@{{{column}}}')],
                formatters={f'@{{{submission_time_col}}}': 'datetime'},
                mode='vline',
                renderers=[glyph_line if len(progress_df) > 1 else glyph_scatter],
            ))
        p.legend.click_policy = "hide"
        return p


def get_colormap(n_cols):
    if n_cols <= 10:
        colormap = all_palettes["Category10"][10][:n_cols]
    elif n_cols <= 20:
        colormap = all_palettes["Category20"][n_cols]
    else:
        colormap = all_palettes["Category20"][20] * int(n_cols / 20 + 1)
        colormap = colormap[:n_cols]
    return colormap
