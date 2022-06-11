from os import times
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import glob
from wordcloud import WordCloud

DEPLOY = False
st.set_page_config(
    page_title="US Baby Names Dashboard",
    page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/282/books_1f4da.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 讀進 csv 檔案
# construct csv file


@st.cache(allow_output_mutation=True)
def construct_csv(path='./Data', deploy=False):
    if deploy:
        all_files = glob.glob(path + r"/*.TXT")
        li = []  # 建立一個空的 list
        for filename in all_files:
            # iterate through each file in the "data" folder that ends with ".TXT"
            df = pd.read_csv(filename, names=[
                'State', 'Gender', 'Year', 'Name', 'No. of Occurrences'])
            li.append(df)

        # concat all the dataframes
        all_df = pd.concat(li, axis=0, ignore_index=True)

        # write out the data to a csv file
        all_df.to_csv(path + '/all_states.csv', index=False)

    # 讀進 csv 檔案
    new_df = pd.read_csv(path + '/all_states.csv')

    return new_df


df = construct_csv(deploy=DEPLOY)


def most_popular_since(year, top: int = 10, states=[]):
    # 先搞定 states 變數
    region_df = df[df.State.isin(states)]

    gender_df = region_df[(region_df['Year'] >= year[0]) & (region_df['Year'] <= year[1])].groupby(['Gender', 'Name']).agg(
        {'No. of Occurrences': 'sum'}).sort_values(by=['Gender', 'No. of Occurrences'])
    gender_df = gender_df.reset_index()

    top_10_F = gender_df[gender_df['Gender'] == 'F'].sort_values(
        by='No. of Occurrences', ascending=False).reset_index(drop=True)[:top]

    top_10_M = gender_df[gender_df['Gender'] == 'M'].sort_values(
        by='No. of Occurrences', ascending=False).reset_index(drop=True)[:top]

    # Girl Plot
    fig_F = px.bar(top_10_F, x='Name', y='No. of Occurrences',
                   title=f'Top {top} Popular Girl Names between {year}')
    fig_F.update_traces(marker_color="#FF66B2")

    # 這個為了美觀留著
    fig_F.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )

    # Boy Plot
    fig_M = px.bar(top_10_M, x='Name', y='No. of Occurrences',
                   title=f'Top {top} Popular Boy Names between {year}')
    fig_M.update_traces(marker_color="#3399FF")
    # 這個為了美觀留著
    fig_M.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )

    return fig_F, fig_M


# Plot for "Map"
# TODO: 註解稍後再加上去
def create_choromap(df, name, gender, begin_year, end_year):
    gen = 'F' if gender == 'female' else 'M'
    df_name_state = df[(df['Name'] == name) & (df['Gender'] == gen)]
    df_filter = df[(df['Gender'] == gen)]
    df_all_name_state = pd.DataFrame(df_filter.groupby(['State', 'Year'])[
                                     'No. of Occurrences'].sum()).reset_index()
    df_all_name_state = df_all_name_state.rename(
        columns={'No. of Occurrences': 'all_count'})
    df_name_state = pd.merge(df_name_state, df_all_name_state, how='outer',
                             left_on=['State', 'Year'], right_on=['State', 'Year'])
    df_name_state['name_pct'] = round(
        df_name_state['No. of Occurrences'] / df_name_state['all_count'] * 100, 3)
    df_name_state['name_pct'] = df_name_state['name_pct'].fillna(0)
    df_name_state = df_name_state[(df_name_state['Year'] <= end_year) & (
        df_name_state['Year'] >= begin_year)]
    df_name_state = df_name_state.sort_values(by=['State', 'Year'])

    df_name_state['text'] = df_name_state['State'] + '<br>' + 'Percentage: ' + df_name_state['name_pct'].astype(
        str) + '%' + '<br>' + 'Count: ' + df_name_state['No. of Occurrences'].astype(str)
    scl_female = [[0.0, 'rgb(224, 224, 224)'], [0.25, 'rgb(255, 204, 229)'],
                  [0.5, 'rgb(255, 102, 178)'], [0.75, 'rgb(255, 0, 127)'],
                  [1.0, 'rgb(102, 0, 51)']]
    scl_male = [[0.0, 'rgb(224, 224, 224)'], [0.25, 'rgb(204, 229, 255)'],
                [0.5, 'rgb(51, 153, 255)'], [0.75, 'rgb(0, 0, 255)'],
                [1.0, 'rgb(0, 0, 102)']]
    scale = scl_female if gender == 'female' else scl_male
    data = [dict(type='choropleth',
                 locations=df_name_state[df_name_state['Year']
                                         == begin_year]['State'],
                 locationmode='USA-states',
                 z=df_name_state[df_name_state['Year'] == begin_year]['name_pct'], text=df_name_state[df_name_state['Year']
                                                                                                      == begin_year]['text'], hoverinfo='text',
                 colorscale=scale, autocolorscale=False,
                 marker=dict(line=dict(color='#FFFFFF', width=2)),
                 colorbar=dict(title='% Named', thickness=15, len=0.6,
                               tickfont=dict(size=14), titlefont=dict(size=14)))]
    layout = dict(title='Percentage of Babies Named ' + name + ' by State from ' + str(begin_year) + ' to ' + str(end_year),
                  geo=dict(scope='usa', showframe=False, showcoastlines=True))
    updatemenus = list([dict(buttons=list()), dict(
        direction='down', showactive=True)])
    years = len(df_name_state['Year'].unique()) + 1
    for n, year in enumerate(df_name_state['Year'].unique()):
        data.append(dict(type='choropleth',
                         locations=df_name_state[df_name_state['Year']
                                                 == year]['State'], locationmode='USA-states',
                         z=df_name_state[df_name_state['Year']
                                         == year]['name_pct'],
                         text=df_name_state[df_name_state['Year']
                                            == year]['text'],
                         hoverinfo='text', colorscale=scale, autocolorscale=False,
                         marker=dict(
                             line=dict(color='#CDEBFF', width=2)),
                         colorbar=dict(title='% Named', thickness=15, len=0.6,
                                       tickfont=dict(size=14), titlefont=dict(size=14)), visible=False))
        visible_traces = [False] * years
        visible_traces[n + 1] = True
        updatemenus[0]['buttons'].append(dict(args=[{'visible': visible_traces}], label=str(year),
                                              method='update'))
    updatemenus[0]['buttons'].append(dict(args=[{'visible': [True] + [False] * (years - 1)}], label='reset',
                                          method='update'))
    layout['updatemenus'] = updatemenus
    fig = dict(data=data, layout=layout)
    return fig


def create_graph_time(df, name, color='rgb(255, 153, 204)', year_range=[1910, 2020]):
    # region_df = df[df.State.isin(states)]
    df_year_pad = pd.DataFrame(
        np.unique(df['Year'])).rename(columns={0: 'Year'})
    df_name = pd.DataFrame(df[df['Name'] == name].groupby('Year')[
                           'No. of Occurrences'].sum()).reset_index()
    df_name_pad = pd.merge(df_year_pad, df_name, how='outer').fillna(0)
    color_tone = color
    graph = go.Scatter(x=df_name_pad['Year'], y=df_name_pad['No. of Occurrences'],
                       line=dict(color=color_tone, width=3), fill='tonexty', name='')
    data = [graph]
    layout = dict(title='Baby Naming Trend for '+name,
                  xaxis=dict(
                      rangeslider=dict(visible=True),
                      range=year_range,
                      title='Year', titlefont=dict(size=16), tickfont=dict(size=13)),
                  yaxis=dict(title='No. of Occurrences',
                             titlefont=dict(size=16), tickfont=dict(size=13)),
                  showlegend=False)
    fig = dict(data=data, layout=layout)
    return fig


def name_diff_plot(year=[1910, 2020], color_most="#FFCC00", color_least="#FFCC00", states=[]):
    region_df = df[df.State.isin(states)]
    df_pre = region_df[region_df['Year'] == year[0]].groupby('Name').agg(
        {'No. of Occurrences': 'sum'}).reset_index().sort_values(by='Name')

    df_post = region_df[region_df.Year == year[1]].groupby('Name').agg(  # 嘗試不同寫法
        {'No. of Occurrences': 'sum'}).reset_index().sort_values(by='Name')
    new_df = pd.merge(df_pre, df_post, on='Name', how='outer').rename(
        columns={'No. of Occurrences_x': 'count_pre', 'No. of Occurrences_y': 'count_post'})
    new_df = new_df.fillna(0)

    new_df['diff'] = new_df['count_post'] - new_df['count_pre']
    new_df['pct_diff'] = new_df['diff'] / new_df['count_pre']

    most_diff_name = new_df[new_df['diff'] ==
                            new_df['diff'].max()].Name.values[0]
    least_diff_name = new_df[new_df['diff'] ==
                             new_df['diff'].min()].Name.values[0]

    fig_most_diff_name = create_graph_time(
        region_df, most_diff_name, color=color_most, year_range=year)
    fig_most_diff_name = go.Figure(fig_most_diff_name)
    fig_most_diff_name.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )

    fig_least_diff_name = create_graph_time(
        region_df, least_diff_name, color=color_least, year_range=year)
    fig_least_diff_name = go.Figure(fig_least_diff_name)
    fig_least_diff_name.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )

    return fig_most_diff_name, fig_least_diff_name, most_diff_name, least_diff_name


def neutral_name(year=[1910, 2020], color_most="#FFCC00", color_least="#FFCC00", states=[]):
    region_df = df[df.State.isin(states)]

    gender_df = region_df[(region_df['Year'] >= year[0]) & (
        region_df['Year'] <= year[1])].groupby(['Name', 'Gender'])['No. of Occurrences'].sum().reset_index()

    gender_df = gender_df.pivot(
        index='Name', columns='Gender', values='No. of Occurrences').reset_index().fillna(0)
    gender_df['diff'] = abs(gender_df['F'] - gender_df['M'])
    gender_df['diff_pct'] = gender_df['diff'] / \
        (gender_df['F'] + gender_df['M'])

    final_df = gender_df[gender_df['diff_pct'] <= 0.1]
    final_df['Total'] = final_df['F'] + final_df['M']

    d = {}
    for a, x in final_df[['Name', 'Total']].values:
        d[a] = x
    wordcloud = WordCloud(background_color='white', width=1600, height=800)
    wordcloud.generate_from_frequencies(frequencies=d)

    # fig = plt.figure(figsize=(16, 8), facecolor=None)
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig, final_df

    # df_pre = region_df[region_df['Year'] == year[0]].groupby('Name').agg(
    #     {'No. of Occurrences': 'sum'}).reset_index().sort_values(by='Name')

    # df_post = region_df[region_df.Year == year[1]].groupby('Name').agg(  # 嘗試不同寫法
    #     {'No. of Occurrences': 'sum'}).reset_index().sort_values(by='Name')
    # new_df = pd.merge(df_pre, df_post, on='Name', how='outer').rename(
    #     columns={'No. of Occurrences_x': 'count_pre', 'No. of Occurrences_y': 'count_post'})
    # new_df = new_df.fillna(0)

    # new_df['diff'] = new_df['count_post'] - new_df['count_pre']
    # new_df['pct_diff'] = new_df['diff'] / new_df['count_pre']

    # most_diff_name = new_df[new_df['diff'] ==
    #                         new_df['diff'].max()].Name.values[0]
    # least_diff_name = new_df[new_df['diff'] ==
    #                          new_df['diff'].min()].Name.values[0]


def main():

    # list of constant variables

    state_lst = sorted(list(df.State.unique()))

    st.write('# US Baby Names Dashboard 👶')
    st.caption('This is USA state-specific data on the relative frequency of given names in the population of U.S. births where the individual has a Social Security Number (Tabulated based on Social Security records as of March 7, 2021)')

    ### Setting Block ###

    container = st.container()
    all_flag = st.checkbox("Select all")

    if all_flag:
        states = container.multiselect('Select Region:',
                                       state_lst, state_lst)
    else:
        states = container.multiselect('Select Region:',
                                       state_lst, ['CA', 'NY'])

    year = st.slider(
        'Select a range of years',
        1910, 2020, (1910, 2020), step=1)
    ### End of Setting Block ###

    st.markdown("***")

    # SECTION 1: Popular Names
    st.subheader('Popular Names from Selected Years', anchor=None)

    top = st.number_input('Top # Names', min_value=1,
                          max_value=50, value=10, step=1)

    col21, col22 = st.columns((5, 5))

    fig_F, fig_M = most_popular_since(
        year=year, top=int(top), states=states)  # 加上 states
    with col21:
        st.plotly_chart(fig_F, use_container_width=True)
    with col22:
        st.plotly_chart(fig_M, use_container_width=True)

    # SECTION 2: Neutral name

    # Plot: Neutral Name Word Cloud
    # Plot: Trend

    st.markdown("***")
    st.subheader('Netural Names', anchor=None)

    col21, _, col22 = st.columns((7, 0.5, 3))
    nm_fig, nm_df = neutral_name(year=year, states=states)
    # buf = BytesIO()
    # nm_fig.savefig(buf, format="png")
    # st.image(buf)
    with col21:
        st.pyplot(nm_fig)
    with col22:
        st.write("#### DF")
        nm_df = nm_df[['Name', 'F', 'M', 'Total']].reset_index(drop=True)
        nm_df[['F', 'M', 'Total']] = nm_df[['F', 'M', 'Total']].astype(int)
        nm_df.columns = ['Name', 'Girl', 'Boy', 'Total Occurrences']
        st.write(nm_df)
    # Plot: Historical Trends of Neutral Names
    # chart_data = pd.DataFrame(np.random.randn(20), columns=['a'])
    # st.caption('Naming Trend for Netural Names', unsafe_allow_html=False)
    # st.line_chart(chart_data)

    # SECTION 3: Largest Increase and Decrease (Most Popular and Unpopular Names)
    # larget decrease and increase from selected period
    st.markdown("***")
    st.subheader('Trends: Most Popular and Unpopular Names', anchor=None)

    fig_most_diff_name, fig_least_diff_name, most_diff_name, least_diff_name = name_diff_plot(
        year=year, color_most="#FFCC00", color_least="#FFCC00", states=states)
    col31, _, col32 = st.columns((5, 0.2, 5))
    with col31:
        st.write(f"#### Most Popular Name {most_diff_name}")
        st.plotly_chart(fig_most_diff_name, use_container_width=True)
    with col32:
        st.write(f"#### Least Popular Name {least_diff_name}")
        st.plotly_chart(fig_least_diff_name, use_container_width=True)

    # SECTION 4: Baby Naming Trends for Specific Names
    st.markdown("***")
    st.subheader('Baby Naming Trends for Specific Names', anchor=None)

    # TODO: Name Trend
    region_df = df[df.State.isin(states)]

    col411, _, col412 = st.columns((5, 0.2, 5))
    with col411:
        input_name = st.text_input('Enter a name:', value='Victoria')

    with col412:
        gender = st.radio(
            "Select a gender to look up:",
            ('female', 'male'))

    col41, _, col42 = st.columns((5, 0.2, 5))
    with col41:
        if gender == "female":
            g_color = "#FF66B2"
        else:
            g_color = "#3399FF"

        specific_name_fig = create_graph_time(
            region_df, input_name, color=g_color, year_range=year)
        specific_name_fig = go.Figure(specific_name_fig)
        specific_name_fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=40),
        )
        st.plotly_chart(specific_name_fig, use_container_width=True)
    with col42:
        # TODO: Map plot
        # QUESTION: 這個有要按照選擇的 states 嗎？
        map_fig = go.Figure(create_choromap(
            df, input_name, gender, year[0], year[1]))
        map_fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=40),
        )
        st.plotly_chart(map_fig, use_container_width=True)


if __name__ == '__main__':
    main()
