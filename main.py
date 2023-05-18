import streamlit as st 
import numpy as np
import pandas as pd 
import plotly.express as px 
from  streamlit_option_menu import option_menu
from PIL import Image
from sklearn.preprocessing import StandardScaler
#st.write("Hello , streamlit world ")

# desplaying text 

#st.text("Text")
#st.write("super function")
#st.header("Header")
#st.subheader("sub-header")
#st.title("Title")
#st.markdown("***markdown***")
#st.code("Print('Hello,World!')",language='python')
#st.latex(r'''
#a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#\sum_{k=0}^{n-1} ar^k =
#a \left(\frac{1-r^{n}}{1-r}\right)
#''')

# desplay button
# btn = st.button("submet")
# if btn:
#     st.info("submeted")

# option = st.radio("Select",['A','B','C'])

# if option == 'A':
#     st.warning("warning!",icon='⚠️')
# elif option == "B":
#     st.error("error",icon='🚨')
# elif option =='C':
#     st.success("Success",icon="✅")

# chk = st.checkbox("I agree")
# if chk:
#     st.exception("Agreement")

# op2 = st.selectbox("select",['A','B','C'])

# if op2 == 'A':
#     st.warning("warning!",icon='⚠️')
# elif op2 == "B":
#     st.error("error",icon='🚨')
# elif op2 =='C':
#     st.success("Success",icon="✅")

# age = st.slider("select",0,100)
# st.write(age)

# st.select_slider("op",[100,200,300])
# st.text_input("Enter a text")
# st.text_area("enter a paragraph")

# st.file_uploader("upload")

# st.camera_input("take a photo") 
# st.date_input("today")

# st.time_input("now")

# st.number_input("num")

# st.multiselect("select",['A','B','C'])

# st.color_picker("colors")

# df = sns.load_dataset('taxis')

# st.write(df)



# btn = st.button("show new sample")
# if btn :
#     st.dataframe(df.sample(5)) # 
# st.table(df.head(2))

#load data 

data = pd.read_csv('financials.csv')
new_data=data.dropna()
new_data=new_data.drop('52 Week Low',axis=1)
new_data=new_data.drop('52 Week High',axis=1)
new_data=new_data.drop('SEC Filings',axis=1)

with st.sidebar: 
    selected = option_menu(
        menu_title="Table of Content",
        options=["About project","Exploratory data analysis","Modeling","Contact us"],
        icons=["bi-card-text","bi-clipboard-data","bi-gear","bi-telephone"],
        #menu_icon = "cast",
        default_index = 0,
      #  orientation = "horizontal",
      )
    
def get_outliers(feautre,data):
    
    no_outliers_df=data.copy()
    
    q1=np.percentile(no_outliers_df[feautre],25)


    q3=np.percentile(no_outliers_df[feautre],75)


    step = (q1-q3)*1.5

    print("Data points considered outliers for the feature '{}':".format(feautre))
    return no_outliers_df[(no_outliers_df[feautre] < q1 - step) & (no_outliers_df[feautre] > q3 + step)]



x=get_outliers('Price/Earnings',new_data)
x=get_outliers('Price/Book',x)
x=get_outliers('Price/Sales',x)
    
X=x[['Price/Sales', 'Price/Book','Price/Earnings']].values
X = StandardScaler().fit_transform(X)
from sklearn.cluster import KMeans

kmeans = KMeans(init= 'k-means++' , random_state=42)
kmeans.fit(X)
y = kmeans.fit_predict(X)

kmeans=KMeans(n_clusters=2 , init='k-means++' , random_state=42)
pred=kmeans.fit_predict(X)

dk=x.copy()
dk['cluster']=pred

    


image = Image.open('stocks.jpg')
if selected == "About project":
    st.markdown("<h1 style='text-align: center; '> Context </h1>", unsafe_allow_html=True)
    st.markdown("**This is a comprehensive dataset including numerous financial metrics that many professionals and investing gurus often use to value companies. This data is a look at the companies that comprise the S&P 500 (Standard & Poor's 500). The S&P 500 is a capitalization-weighted index of the top 500 publicly traded companies in the United States (top 500 meaning the companies with the largest market cap). The S&P 500 index is a useful index to study because it generally reflects the health of the overall U.S. stock market. The dataset was last updated in July 2020.**")
    st.subheader("Objective")
    st.markdown("**clustring the stocks to 2 clusters, then classfiy them undervalue or overvalue**")
    st.image(image)
    st.warning("Please read the following to understand the project",icon='⚠️')
    st.markdown("* Price: the stock price. ")
    st.markdown("* Earnings/Share: Earnings per share or EPS is an important financial measure, which indicates the profitability of a company. ")
    st.markdown("* Price/Earnings: The Price Earnings Ratio (P/E Ratio) is the relationship between a company’s stock price and earnings per share (EPS). It is a popular ratio that gives investors a better sense of the value of the company. ")
    st.markdown("* Dividend yield: is a ratio, and one of several measures that helps investors understand how much return they are getting on their investment. For companies that pay a dividend.")
    st.markdown("* Market capitalization: the total value of all a company's shares of stock.")
    st.markdown("* EBITDA: is an alternate measure of profitability to net income,EBITDA attempts to represent cash profit generated by the company’s operations. ")
    st.markdown("* Price/Sales: The price-to-sales (P/S) ratio compares a company's stock price to its revenues, helping investors find undervalued stocks that make good investments.")
    st.markdown("* Price/Book: Many investors use the price-to-book ratio (P/B ratio) to compare a firm's market capitalization to its book value and locate undervalued companies.")
    reading = st.checkbox('I have read ')
    if reading:
        st.success("Thank you, enjoy your journey ",icon='✅')
    
if selected == "Exploratory data analysis":
    st.markdown("<h1 style='text-align: center; '>Exploratory data analysis</h1>", unsafe_allow_html=True)

    st.header("Detecting outliers for Price/Sales")
    fig = px.box(new_data, y="Price/Sales",color='Sector',hover_data=['Symbol','Name'])
    st.plotly_chart(fig)

    st.header("Detecting outliers for Price/Book")
    fig = px.box(new_data, y="Price/Book",color='Sector',hover_data=['Symbol','Name'])
    st.plotly_chart(fig)
    st.header("Detecting outliers for Price/Earning")
    fig = px.box(new_data, y="Price/Earnings",color='Sector',hover_data=['Symbol','Name'])
    st.plotly_chart(fig)
    st.markdown("* there are outliers")
    st.header("Highest market cap companies")
    ranked_market_cap = data.copy()
    ranked_market_cap.sort_values('Market Cap',inplace=True,ascending=False)
    fig = px.bar(ranked_market_cap.head(10), y='Market Cap',x="Name",color='Sector',title='Top 10 market cap stocks valuation',template = "plotly_white")
    st.plotly_chart(fig)
    st.markdown("* Apple is the highest market cap is 809B, it's normal because the technology sector is the highest one.") 
    
    st.header("P/E vs Price")
    fig = px.scatter(data, x = "Price", y = "Price/Earnings", color='Sector', template = "plotly_white",hover_data=['Symbol','Name'])
    fig.update_layout(yaxis_range=[0,50])
    fig.update_layout(xaxis_range=[0,500])
    st.plotly_chart(fig)
    st.markdown("* A high PE ratio means that a stock is expensive and its price may fall in the future. A low PE ratio means that a stock is cheap and its price may rise in the future. The PE ratio, therefore, is very useful in making investment decisions.")


    st.header("Dividend Yield VS price")
    fig = px.scatter(data, x="Price", y="Dividend Yield",facet_col="Sector",facet_col_wrap=3 ,color='Sector')
    fig.update_layout(yaxis_range=[-0.5,8])
    fig.update_layout(xaxis_range=[0,400])
    st.plotly_chart(fig)
    st.markdown("* High dividend yield stocks indicate how much a firm pays out in dividends about its market share price each year. and that influence the growth of the company negatively.")
    st.markdown("* *Example of Dividend Yield* Suppose Company A’s stock is trading at 20 and pays annual dividends of 1 per share to its shareholders. Suppose that Company B's stock is trading at 40 and also pays an annual dividend of 1 per share. This means Company A's dividend yield is 5% (1 / 20), while Company B's dividend yield is only 2.5 (1 / 40). Assuming all other factors are equivalent, an investor looking to use their portfolio to supplement their income would likely prefer Company A over Company B because it has double the dividend yield. ")
    

    st.header("Dividend Yield VS Market cap")
    fig = px.density_heatmap(data, x="Market Cap", y="Dividend Yield",color_continuous_scale='Tropic')
    st.plotly_chart(fig)
    fig = px.density_heatmap(data, x="Market Cap", y="Dividend Yield",color_continuous_scale='Tropic',facet_col="Sector",facet_col_wrap=3)
    st.plotly_chart(fig)
    st.markdown("* when dividend yeld decrease markat cap also decrease.")
    

    st.header("EBITDA% per sector")
    fig = px.pie(data, values='EBITDA', names='Sector')
    st.plotly_chart(fig)
    st.markdown("* Information technology is the highest")
    st.header("correlation")
    fig = px.imshow(new_data['Price','Price/Earnings','Dividend Yield','Earnings/Share','Market Cap','EBITDA','Price/Sales','Price/Book'].corr(),text_auto=True,aspect="auto")
    st.plotly_chart(fig)
if selected == "Modeling":
    st.title(f"{selected}") 
    st.header("Missing Value")
    fig = px.imshow(data.isnull(),aspect="auto")
    st.plotly_chart(fig)
    st.markdown("* all missing data have been removed because they less than 1% of the data ")

    st.header("Clustring")
    fig = px.scatter_3d(dk, x='Price/Sales', y='Price/Book', z='Price/Earnings',
              color='cluster')
    st.plotly_chart(fig)
    st.markdown("* here 2 cluster undervalue and overvalue")
    
    st.subheader("Test the model ")
    price = st.number_input('Price')
    pe = st.number_input('Price/Earnings')
    dy = st.number_input('Dividend Yield')
    mc = st.number_input('Market Cap')
    eps=st.number_input('Earnings/Share')
    eb = st.number_input('EBITDA')
    ps = st.number_input('Price/Sales')
    pb = st.number_input('Price/Book')
    btn = st.button("submet")
    if btn:





        
        classi_data=dk[['Price','Price/Earnings','Dividend Yield','Earnings/Share','Market Cap','EBITDA','Price/Sales','Price/Book','cluster']]

        x = classi_data.drop("cluster" , axis = 1)
        y = classi_data['cluster']

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x ,y ,test_size=0.20, random_state=27)
       
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5, max_features=4, n_estimators=100)
        clf.fit(x_train , y_train)

        features = np.array([[price,pe,dy,eps,mc,eb,ps,pb]])
        if 0 in clf.predict(features):
            st.info("The stock is Overvalue")
        else:
            st.info("The stock is undervalue")
    st.warning("The model is under development, please do Not consider this a financial advice ",icon="⚠️")

    st.info("Model V2 will contain more features ,more accurate and Supervised tests under people has domain expertise")
        
if selected == "Contact us":
    st.title(f"{selected}") 
    st.subheader("Linked in")
    st.info("https://www.linkedin.com/in/ziad-ehab-953183244",icon="🧑")
    st.subheader("Number")
    st.info("01206455185",icon='📞')
    st.subheader("Email")
    st.info("Ziad.ehab15@yahoo.com",icon='📧')
   
