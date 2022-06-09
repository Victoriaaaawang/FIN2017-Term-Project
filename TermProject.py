from os import times
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio

# 讀進 csv 檔案
df = pd.read_csv("C:/Users/Victoria Wang/Desktop/NTU/110-2/Programming/Data/all_states.csv")



def most_popular_since(year, top: int = 10, states = []):
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
    fig_F.update_traces(marker_color='pink')
    # 這個為了美觀留著
    fig_F.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )
    
    # Boy Plot
    fig_M = px.bar(top_10_M, x='Name', y='No. of Occurrences',
                    title=f'Top {top} Popular Boy Names between {year}')
    fig_M.update_traces(marker_color='cornflowerblue')
    # 這個為了美觀留著
    fig_M.update_layout(
        margin=dict(l=0, r=0, t=40, b=40),
    )

    return fig_F, fig_M


def most_popular_in(year):
    pass
    # 'Year' == Year


# Plot for "Map"
# TODO: 註解稍後再加上去
def create_choromap(df, name, gender, begin_year, end_year):
    gen = 'F' if gender == 'female' else 'M'
    df_name_state = df[(df['Name'] == name) & (df['Gender'] == gen)]
    df_filter = df[(df['Gender'] == gen)]
    df_all_name_state = pd.DataFrame(df_filter.groupby(['State', 'Year'])['No. of Occurrences'].sum()).reset_index()
    df_all_name_state = df_all_name_state.rename(columns={'No. of Occurrences': 'all_count'})
    df_name_state = pd.merge(df_name_state, df_all_name_state, how='outer', 
                             left_on=['State', 'Year'], right_on=['State', 'Year'])
    df_name_state['name_pct'] = round(df_name_state['No. of Occurrences'] / df_name_state['all_count'] * 100, 3)
    df_name_state['name_pct'] = df_name_state['name_pct'].fillna(0)
    df_name_state = df_name_state[(df_name_state['Year'] <= end_year) & (df_name_state['Year'] >= begin_year)]
    df_name_state = df_name_state.sort_values(by=['State', 'Year'])

    df_name_state['text'] = df_name_state['State'] + '<br>' + 'Percentage: ' + df_name_state['name_pct'].astype(str) + '%' + '<br>' + 'Count: ' + df_name_state['No. of Occurrences'].astype(str) 
    scl_female = [[0.0, 'rgb(224, 224, 224)'], [0.25, 'rgb(255, 204, 229)'],
                  [0.5, 'rgb(255, 102, 178)'], [0.75, 'rgb(255, 0, 127)'],
                  [1.0, 'rgb(102, 0, 51)']]
    scl_male = [[0.0, 'rgb(224, 224, 224)'], [0.25, 'rgb(204, 229, 255)'],
                [0.5, 'rgb(51, 153, 255)'], [0.75, 'rgb(0, 0, 255)'],
                [1.0, 'rgb(0, 0, 102)']]
    scale = scl_female if gender == 'female' else scl_male
    data = [dict(type='choropleth',
                 locations=df_name_state[df_name_state['Year']==begin_year]['State'],
                 locationmode='USA-states', 
                 z=df_name_state[df_name_state['Year']==begin_year]['name_pct'], text=df_name_state[df_name_state['Year']==begin_year]['text'], hoverinfo='text',
                 colorscale=scale, autocolorscale=False,
                 marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                 colorbar=dict(title='% Named', thickness=15, len=0.6,
                               tickfont=dict(size=14), titlefont=dict(size=14)))]
    layout = dict(title='Percentage of Babies Named ' + name + ' by State from ' + str(begin_year) + ' to ' + str(end_year),
                  font=dict(size=14),
                  geo=dict(scope='usa', showframe=False, showcoastlines=True))
    updatemenus = list([dict(buttons=list()), dict(direction='down', showactive=True)])
    years = len(df_name_state['Year'].unique()) + 1
    for n, year in enumerate(df_name_state['Year'].unique()):
        data.append(dict(type='choropleth',
                    locations=df_name_state[df_name_state['Year']==year]['State'], locationmode='USA-states', 
                    z=df_name_state[df_name_state['Year']==year]['name_pct'],
                    text=df_name_state[df_name_state['Year']==year]['text'],
                    hoverinfo='text', colorscale=scale, autocolorscale=False,
                    marker=dict(line=dict(color='rgb(255,255,255)', width=2)),
                    colorbar=dict(title='% Named', thickness=15, len=0.6,
                                  tickfont=dict(size=14), titlefont=dict(size=14)), visible=False))
        visible_traces = [False] * years
        visible_traces[n + 1] = True
        updatemenus[0]['buttons'].append(dict(args=[{'visible': visible_traces}], label=str(year),
                                              method='update'))
    updatemenus[0]['buttons'].append(dict(args=[{'visible': [True] + [False] *  (years - 1)}], label='reset',
                                          method='update'))
    layout['updatemenus'] = updatemenus
    fig = dict(data=data, layout=layout)
    return fig


# Plot for "Popular Names"



def main():
    st.set_page_config(
        page_title="Baby Names Demo",
        page_icon="https://emojipedia-us.s3.dualstack.us-west-1.amazonaws.com/thumbs/240/twitter/282/books_1f4da.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # list of constant variables

    state_lst = sorted(list(df.State.unique()))
    
    st.write('# US Baby Names Dashboard 👶')
    st.caption('This is the baby birth names of United States from 1910 to 2021.')


    ### Region Block ###

    container = st.container()
    all_flag = st.checkbox("Select all")
    
    if all_flag:
        states = container.multiselect('Select Region:',
            state_lst, state_lst)
    else:
        states =  container.multiselect('Select Region:',
            state_lst)

    ### End of Region Block ###

    st.write(df.head())
    

    st.markdown("***")

    st.subheader('Popular Names from Selected Years', anchor=None)
    st.write(f"""States Selected: {states}""")

    col11, col121, col12 = st.columns((5, 0.2, 5)) # col121 是為了讓 range 不要 overlap 到 col12
    with col11:
        year = st.slider(
            'Select a range of years',
            1910, 2021, (1910, 2021), step=1)
        # year = st.number_input('Popular Names Since ... Year',
        #                        min_value=1910, max_value=2021, value=1910, step=1)
    with col12:
        top = st.number_input('Top # Names', min_value=1,
                              max_value=50, value=10, step=1)

    col21, col22 = st.columns((5, 5))

    fig_F, fig_M = most_popular_since(year=year, top=int(top), states=states) # 加上 states
    with col21:
        st.plotly_chart(fig_F, use_container_width=True)
    with col22:
        st.plotly_chart(fig_M, use_container_width=True)
    
    # TODO: Map plot 
    # QUESTION: 這個有要按照選擇的 states 嗎？


    input_name = st.text_input('Enter a name:', value='Victoria')
    map_fig = go.Figure(create_choromap(df, input_name, 'female', 2005, 2015))
    # pio.show(map_fig)
    # # st.write(map_fig)
    st.plotly_chart(map_fig, use_container_width=True)

    # TODO: Name Trend
    

    

if __name__ == '__main__':
    main()



# Plot2: Neutral Name Word Cloud

# import streamlit_wordcloud as wordcloud
st.markdown("***")
st.subheader('Netural Names', anchor=None)





# Plot3: Historical Trends of Neutral Names 
chart_data = pd.DataFrame( np.random.randn(20), columns=['a'])
st.caption('Naming Trend for Netural Names', unsafe_allow_html=False)
st.line_chart(chart_data)


### larget decrease and increase from selected period
st.markdown("***")
st.subheader('Trends: Most Popular and Unpopular Names', anchor=None)

#### Names with largest decrease and increase in number since 1980

# max and min of (Name in X year - Name in Y Year) #先不分姓別
# Mary 1990 和 Mary 2010年之間 數量差異



def most_pop_unpop(year, states = []):
    region_df = df[df.State.isin(states)]

    year_df = region_df[(region_df['Year'] >= year[0]) & (region_df['Year'] <= year[1])].groupby(['Name']).agg(
            {'No. of Occurrences': 'sum'}).sort_values(by=['Year', 'No. of Occurrences'])
    year_df = year_df.reset_index()

    delta_df = year_df[['Name'][year_df['Year']== year[1]]] - year_df[year_df['Year']== year[0]].sort_values(
        by='No. of Occurrences', ascending=False).reset_index()

    
    inc_most = delta_df['No. of Occurrences'] 
    dec_most = delta_df['No. of Occurrences']


    col31, col32 = st.columns((5, 5))
    with col31:
        Fig_inc = st.metric(label="Most Popular Name", value= inc_most, delta=10,
                    delta_color="inverse")
    with col32:
        Fig_dec = st.metric(label="Most Unpopular Name", value= dec_most, delta=-0.5,
                    delta_color="inverse")

    col41, col42 = st.columns((5, 5))
    with col41:
        chart_data = pd.DataFrame( np.random.randn(20), columns=['Name'])
    with col42:
        chart_data = pd.DataFrame( np.random.randn(20), columns=['Name'])
    with col41:
        st.line_chart(chart_data)
    with col42:
        st.line_chart(chart_data)



### Insert Name


st.markdown("***")
st.subheader('Baby Naming Trends for Specific Names', anchor=None)
input_name = st.text_input('Name')
st.write('Search for', input_name)



# backup
# def create_graph_time_backup(df, name, gender):
#     df_year_pad = pd.DataFrame(np.unique(df['Year'])).rename(columns={0: 'Year'})
#     df_name = pd.DataFrame(df[df['Name'] == name].groupby('Year')['No. of Occurrences'].sum()).reset_index()
#     df_name_pad = pd.merge(df_year_pad, df_name, how='outer').fillna(0)
#     color_tone = 'rgb(255, 153, 204)' if gender == 'female' else 'rgb(102, 178, 255)'
#     graph = go.Scatter(x=df_name_pad['Year'], y=df_name_pad['No. of Occurrences'],
#                        line=dict(color=color_tone, width=3), fill='tonexty', name='')
#     line = go.Scatter(x=df_name_pad['No. of Occurrences'].max().astype(int),
#                       y=[i for i in range(df_name_pad['No. of Occurrences'].max().astype(int))],
#                       line=dict(color=color_tone, width=3, dash='dot'),
#                       hoverinfo='none')
#     data = [graph, line]
#     layout = dict(title='Baby Naming Trend for '+name, titlefont=dict(size=22),
#                   xaxis=dict(rangeslider=dict(visible=True),
#                              title='Year', titlefont=dict(size=16), tickfont=dict(size=13)),
#                   yaxis=dict(title='No. of babies named',
#                              titlefont=dict(size=16), tickfont=dict(size=13)),
#                   showlegend=False)
#     fig = dict(data=data, layout=layout)
#     return fig

# fig_name = create_graph_time_backup(df, input_name, 'female')
# py.iplot(fig_name)
