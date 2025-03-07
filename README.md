<h1 align="center">Hi 👋, I'm Akshith Reddy</h1>
<h3 align="center">A passionate Programmer from India</h3>

<p align="left"> <img src="https://komarev.com/ghpvc/?username=speedbird22&label=Profile%20views&color=0e75b6&style=flat" alt="speedbird22" /> </p>

- 🌱 I’m currently learning **Java, Python**

- 💬 Ask me about **HTML, CSS, C++**

- 📫 How to reach me **akshithreddyworld2020@gmail.com**

- ⚡ Fun fact **HTML is my most favorite language**

<h3 align="left">Connect with me:</h3>
<p align="left">
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.arduino.cc/" target="_blank" rel="noreferrer"> <img src="https://cdn.worldvectorlogo.com/logos/arduino-1.svg" alt="arduino" width="40" height="40"/> </a> <a href="https://www.w3schools.com/cpp/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/cplusplus/cplusplus-original.svg" alt="cplusplus" width="40" height="40"/> </a> <a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> </a> <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> </a> <a href="https://www.java.com" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/java/java-original.svg" alt="java" width="40" height="40"/> </a> <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> </a> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>



<!DOCTYPE html>
<html>
<head>
    body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        code {
            background: #f4f4f4;
            padding: 5px;
            border-radius: 5px;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
        }
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
    </style>
</head>

    
<body>
    <h1>Amazon Market Basket Analysis - Data Mining Year 1 Summative Assessment</h1>
        <h2>Project Overview</h2>
    <p>This project analyzes Amazon's e-commerce data to uncover customer shopping behaviors, segment customers, and identify relationships between products using data mining techniques.</p>
        <h2>Project Scope</h2>
    <ul>
        <li>Perform <strong>customer segmentation</strong> using clustering techniques.</li>
        <li>Identify <strong>frequent itemsets</strong> using association rule mining.</li>
        <li>Analyze <strong>user behavior</strong> to enhance marketing strategies.</li>
        <li>Develop an <strong>interactive Streamlit dashboard</strong> to present findings.</li>
    </ul>

   <h2>Dataset & Preprocessing</h2>
    <ul>
        <li>Dataset includes attributes: <code>Product ID, Category, Actual Price, Discounted Price, Rating, Reviews</code>.</li>
        <li>Handled missing values, removed outliers, and normalized numerical attributes.</li>
        <li>Encoded categorical data using one-hot encoding.</li>
    </ul>

  <h2>Exploratory Data Analysis (EDA)</h2>
  <ul>
        <li>Plotted <strong>histograms</strong> and <strong>boxplots</strong> for price distributions.</li>
        <li>Visualized <strong>category distributions</strong> with bar charts.</li>
        <li>Created <strong>heatmaps</strong> to analyze correlations between pricing, ratings, and categories.</li>
    </ul>

<h2>Customer Segmentation (Clustering)</h2>
    <ul>
        <li>Applied <strong>K-Means Clustering</strong> and <strong>Hierarchical Clustering</strong> to group customers.</li>
        <li>Features used: <code>Discounted Price, Actual Price, Product Category, Rating</code>.</li>
        <li>Identified customer segments for personalized marketing.</li>
    </ul>

 <h2>Market Basket Analysis (Association Rule Mining)</h2>
    <ul>
        <li>Used <strong>Apriori Algorithm</strong> to identify frequent product associations.</li>
        <li>Generated <strong>Support, Confidence, and Lift</strong> metrics for rule validation.</li>
        <li>Discovered product bundling opportunities.</li>
    </ul>

  <h2>User Behavior Analysis</h2>
    <ul>
        <li>Performed <strong>sentiment analysis</strong> on customer reviews using NLP.</li>
        <li>Generated <strong>word clouds</strong> for review trends.</li>
    </ul>

 <h2>Deployment with Streamlit</h2>
    <ul>
        <li>Developed an interactive <strong>dashboard</strong> to visualize findings.</li>
        <li>Includes features for customer segmentation, frequent itemsets, and sentiment analysis.</li>
        <li>Deployed via <strong>Streamlit Cloud</strong>.</li>
    </ul>

  <h2>GitHub Repository Contents</h2>
    <ul>
        <li><code>notebooks/</code> - Jupyter Notebook with analysis and visualizations.</li>
        <li><code>data/</code> - Cleaned dataset used in the project.</li>
        <li><code>streamlit_app.py</code> - Streamlit deployment script.</li>
        <li><code>README.html</code> - This documentation.</li>
    </ul>

<h2>How to Run the Project</h2>
    <pre><code>pip install -r requirements.txt
streamlit run streamlit_app.py</code></pre>

  <h2>Project Links</h2>
    <ul>
        <li><strong>GitHub Repository:</strong> <a href="https://github.com/YourUsername/YourRepository">Click Here</a></li>
        <li><strong>Streamlit App:</strong> <a href="https://share.streamlit.io/YourApp">Click Here</a></li>
    </ul>

  <h2>Contributors</h2>
    <p><strong>Student Name:</strong> Your Name<br>
    <strong>Candidate Registration Number:</strong> XXXXXXXX<br>
    <strong>CRS Name:</strong> Artificial Intelligence<br>
    <strong>Course Name:</strong> Data Mining<br>
    <strong>School Name:</strong> Your Institution</p>
</body>
</html>

