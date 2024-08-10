import streamlit as st
import preprocess
import re
import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

st.title('ChatWhiz: Whatsapp Chat Analyzer')
st.sidebar.title("Upload you chat")

# this is for uploading a file

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    # converting the bytecode to the text-file

    data = bytes_data.decode("utf-8")

    # sending the file data to the preprocess function for further functioning

    df = preprocess.preprocess(data)

    # displaying the dataframe

    # st.dataframe(df)

    # fetch unique users
    user_list = df['User'].unique().tolist()

    # removing the groupnotification
    user_list.remove('Group Notification')

    # organinsing things
    user_list.sort()

    # including overall,this will be responsible for showcasing the  overall chat group analysis

    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox(
        "Show analysis with respect to", user_list)

    
    if st.sidebar.button("Show Analysis"):
        st.markdown(f"<h2>Whats App Chat Analysis for  {selected_user}</h2>",unsafe_allow_html=True)

        # getting the stats of the selected user from the stats script

        num_messages, num_words, media_omitted, links = stats.fetchstats(selected_user, df)

        # first phase is to showcase the basic stats like number of users,number of messages,number of media shared and all,so for that i requrire the 4 columns

        card_template = """
                <style>
                
                a svg {{
                    display:none;
                }}
                
                .card {{
                    background-color: rgba(128, 128, 128, 0.2); /* Transparent grey */
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    margin-bottom: 40px;
                    transition: transform 0.2s, box-shadow 0.2s;
                    border: 1px solid transparent; /* Initial border */
                }}

                .card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0px 8px 12px rgba(0, 255, 0, 0.5); /* Green glow */
                    border-color: rgba(0, 255, 0, 0.5); /* Green border on hover */
                    border-width: 2px; /* Set the width of the border */
                    border-style: solid; /* Solid border style */
                }}


                .card h3 {{
                    color: #ffffff;
                }}

                .card h1 {{
                    color: #0df005;
                }}
                </style>

                <div class="card">
                    <h3>{header}</h3>
                    <h1>{title}</h1>
                </div>

                """

        # Create columns
        col1, col2 = st.columns(2)

        # Display cards in each column
        with col1:
            st.markdown(card_template.format(header="Total Messages", title=num_messages), unsafe_allow_html=True)

        with col2:
            st.markdown(card_template.format(header="Total No. of Words", title=num_words), unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(card_template.format(header="Media Shared", title=media_omitted), unsafe_allow_html=True)

        with col4:
            st.markdown(card_template.format(header="Total Links Shared", title=links), unsafe_allow_html=True)

        # finding the busiest users in the group

        if selected_user == 'Overall':

            # dividing the space into two columns
            # first col is the bar chart and the second col is the dataframe representing the

            st.title('Most Busy Users')
            busycount, newdf = stats.fetchbusyuser(df)

            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
            # Create a Plotly pie chart
            fig = go.Figure(data=[go.Pie(
                labels=busycount.index,
                values=busycount.values,
                hole=0.5,
                textinfo='label+percent',
                insidetextorientation='radial',
                marker=dict(colors=colors),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent:.1%}<extra></extra>',

            )])

            # Update layout to match your style
            fig.update_layout(
                margin=dict(t=50, b=50, l=50, r=50),
                paper_bgcolor='rgba(128, 128, 128,0.2)',
                # plot_bgcolor='rgba(128, 255, 255, 0.2)',
                hoverlabel=dict(bgcolor="Green", font_size=16, font_family="Rockwell"),
                legend=dict(orientation="h", xanchor="center", x=0.5, y=-0.2),
            )

            # Display the Plotly figure in Streamlit
            st.plotly_chart(fig)

        # Word Cloud
        plot_bg_color='#4e524e'
        st.title('Word Cloud')
        df_img = stats.createwordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_img)
        ax.axis('off')  # Hide the axes

        # Update layout colors
        fig.patch.set_facecolor(plot_bg_color)  # Background color of the figure
        ax.set_facecolor(plot_bg_color)  # Background color of the plotting area

        st.pyplot(fig)

        # most common words in the chat

        most_common_df = stats.getcommonwords(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1],color='Green')
        plt.xticks(rotation='vertical')
        st.title('Most commmon words')
        st.pyplot(fig)

        # Emoji Analysis

        emoji_df = stats.getemojistats(selected_user, df)
        # emoji_df.columns = ['Emoji', 'Count']
        st.title("Emoji Analysis")
        if not emoji_df.empty and emoji_df.shape[0] > 0:
            # Rename columns if needed
            emoji_df.columns = ['Emoji', 'Count']

            # Calculate percentage use
            emojicount = list(emoji_df['Count'])
            perlist = [(i / sum(emojicount)) * 100 for i in emojicount]
            emoji_df['Percentage use'] = np.array(perlist)

            # Display title and DataFrame
            # st.title("Emoji Analysis")
            st.dataframe(emoji_df)
        else:
            st.write("No emoji data available for the selected user.")
        # Monthly timeline

        st.title("Monthly Timeline")
        time = stats.monthtimeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(time['Time'], time['Message'], color='green')
        plt.xticks(rotation='vertical')
        plt.tight_layout()
        st.pyplot(fig)

        # Activity maps

        st.title("Activity Maps")

        col1, col2 = st.columns(2)

        with col1:

            st.header("Most Busy Day")

            busy_day = stats.weekactivitymap(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='Green')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)

        with col2:

            st.header("Most Busy Month")
            busy_month = stats.monthactivitymap(selected_user, df)

            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='Yellow')
            plt.xticks(rotation='vertical')
            plt.tight_layout()
            st.pyplot(fig)


    
    if st.sidebar.button("Show Sentiment Analysis", key='SA'):
        overall_sentiment, sentiment_counts = stats.Sentiment_analysis(selected_user, df)
        st.markdown(f"""<h2>Sentiment Analysis of {selected_user}</h2>
                    <span style="cursor: pointer; color: blue;" 
                    title="The Overall Sentiment is calculated by the majority of messages of the sentiments,
                    and if two values are equal then weighted score is used">ℹ️</span>""", unsafe_allow_html=True)
        
        # st.write(f"Overall Sentiment: **{overall_sentiment}**")
        # st.write(f"Positive: {sentiment_counts['POSITIVE']} messages")
        # st.write(f"Negative: {sentiment_counts['NEGATIVE']} messages")
        # st.write(f"Neutral: {sentiment_counts['NEUTRAL']} messages")
        
        if overall_sentiment == 'Positive':
            background_color = '#4CAF50'
            shadow_color = '#388E3C'
        elif overall_sentiment == 'Negative':
            background_color = '#f44336'
            shadow_color = '#d32f2f'
        else:
            background_color = '#FFC107'
            shadow_color = '#FFA000'
            # box-shadow: 0px 4px 8px {shadow_color};

        
        st.markdown(
            f"""
            <div style='background-color:{background_color}; padding: 20px; border-radius: 10px;
            text-align: center;
            margin-bottom:20px'>
            <h3>Overall Sentiment</h3>
            <p style='font-size: 24px; font-weight: bold;'>{overall_sentiment}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)


        # Card 2: Positive Sentiment
        with col1:
            st.markdown(
                f"""
                <div style='background-color:#2196F3; padding: 20px; border-radius: 10px;
                text-align: center;
                margin-bottom:20px'>
                    <h3>Positive</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{sentiment_counts['POSITIVE']} messages</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Card 3: Negative Sentiment
        with col2:
            st.markdown(
                f"""
                <div style='background-color:#f44336; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3>Negative</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{sentiment_counts['NEGATIVE']} messages</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        col1, col2, col3= st.columns(3)

        # Card 4: Neutral Sentiment
        with col2:
            st.markdown(
                f"""
                <div style='background-color:#FFC107; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3>Neutral</h3>
                    <p style='font-size: 24px; font-weight: bold;'>{sentiment_counts['NEUTRAL']} messages</p>
                </div>
                """,
                unsafe_allow_html=True
            )
                