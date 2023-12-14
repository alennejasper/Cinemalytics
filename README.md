<a name = "readme-ceiling" id = "readme-ceiling"></a>

<div align = "center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

[contributors-shield]: https://img.shields.io/github/contributors/alennejasper/Cinemalytics.svg?style=for-the-badge
[contributors-url]: https://github.com/alennejasper/Cinemalytics/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/alennejasper/Cinemalytics-.svg?style=for-the-badge
[forks-url]: https://github.com/alennejasper/Cinemalytics/network/members

[stars-shield]: https://img.shields.io/github/stars/alennejasper/Cinemalytics.svg?style=for-the-badge
[stars-url]: https://github.com/alennejasper/Cinemalytics/stargazers

[issues-shield]: https://img.shields.io/github/issues/alennejasper/Cinemalytics-.svg?style=for-the-badge
[issues-url]: https://github.com/alennejasper/Cinemalytics/issues

</div>


<br/>


<div align = "center">

<a href = "https://github.com/alennejasper/Cinemalytics">
<img src = "static/icons/icon.png" alt = "Logo" height = "100">
</a>

<br/>

<h3 align = "center">
<b>Beyond the Screen:</b>
<br/>
<b>The Exploration with Cinemalytics to Unveil Box Office Big Data</b>
</h3>

<br/>

<p align = "center">
<i>Cinemalytics is a sophisticated web application tailored for Big Data processing in the realm of the box office industry that offers a user-friendly interface. Accordingly, the platform will help enable users to explore both descriptive and predictive analytics with a tap into extensive box office dataset.</i>

<br/>

<a href = "https://github.com/alennejasper/Cinemalytics"><b>👀 Explore the Documentation 🔭</b></a>

<br/>

🎬
<a href = "https://github.com/alennejasper/Cinemalytics">Witness the Project</a>
🍿
<a href = "https://github.com/alennejasper/Cinemalytics/issues">Report a Bug</a>
📝
<a href = "https://github.com/alennejasper/Cinemalytics/issues">Request a Feature</a>
🎥

</p>

</div>


<br/>


<details align = "justify">

<summary><b>Table of Contents</b></summary>
  
1. [Project Description](#project-description)

<br/>

2. [Implemented Features](#implemented-features)

<br/>

3. [Deployment Instructions](#deployment-instructions)

<br/>

4. [Project Specifications](#project-specifications)

<br/>

5. [Software License](#software-license)

<br/>

6. [Acknowledgements](#acknowledgements)

</details>


<br/>


### *_Project Description_*

<p align = "justify">

The box office analytics project, Cinemalytics, unveils a meticulously developed box office web application alongside the backend prowess of Django, with an unwavering focus on Big Data approaches and SQLite3 as the database foundation. 

<br/>

Dive into the core of this project, wherein the primary goal transcends conventional boundaries, which aims to construct a comprehensive box office analytics project that goes further with the integration of Big Data processing techniques to build a scalable box office prediction model and implement visualization charts for thorough data insights. 

<br/>

</p>


<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>


<br/>


### *_Implemented Features_*

<p align = "justify">

<b>1. Big Data Processing</b>
> Utilizes the Big Data processing techniques for an efficient management of large–scale box office information. 

<br/>

<b>2. Data Cleansing</b>
> Implements thorough cleansing of box office data in order to leverage the capabilities of Big Data processing.

<br/>

<b>3. Descriptive Analytics</b>
> Incorporates stellar functions for quick and efficient retrieval of summary statistics, key metrics, and data distribution.

<br/>

<b>4. Predictive Analytics</b>
> Constructs a multiple linear regression model to predict box office performance based on extensive box office features.

<br/>

<b>5. Data Visualization</b>
> Utilizes the bell curve, line graph, histogram, radar chart, scatter plot, and correlation matrix based on box office features to provide comprehensive insights.

<br/>

</p>


<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>


<br/>


### *_Deployment Instructions_*

<p align = "justify">

<b>1. Clone the Repository</b>
> Begin by cloning the Cinemalytics repository to your local machine using the following command.

<br/>

```
git clone https://github.com/alennejasper/Cinemalytics.git
``` 

<br/>

<b>2. Navigate to the Cinemalytics Directory</b>
> Move into the Cinemalytics directory.

<br/>

```
cd Cinemalytics
```

<br/>

<b>3. Install Python</b>
> Ensure Python 3.10.2 is installed on your system. If not, you can download and install it from <a href = "https://www.python.org/downloads/">python.org.</a>

<br/>

```
python --version
```

<br/>

<b>4. Install Pipenv</b>
> Ensure Pipenv is installed on your system. If not, you can install it using the code below.

<br/>

```
pip install pipenv.
```

<br/>

<b>5. Install Django</b>
> Install Django using the following command.

<br/>

```
pip install django.
```

<br/>

<b>6. Install Dependencies</b>
> Install dependencies from the requirements.txt file using Pipenv by running this in the Cinemalytics directory.

<br/>

```
pip install -r requirements.txt
```

<br/>

<b>7. Activate the Virtual Environment</b>
> Activate the virtual environment using Pipenv.

<br/>

```
pipenv shell
```

<br/>

<b>8. Apply Migrations</b>
> Apply any necessary database migrations.

<br/>

```
python manage.py migrate
```

<br/>

<b>9. Run the Development Server</b>
> Start the development server using the following command. 

<br/>

```
python manage.py runserver
```

<br/>

<b>10. Access the Application</b>
> Once the server is running, open your web browser and then visit the local host address.

<br/>

```
http://127.0.0.1:8000/
```

<br/>

<b>11. Explore and Contribute</b>
> The application is now up and running! Feel free to explore the box office features, and do not hesitate to delve into the codebase. When you are done, use Ctrl+C command in the terminal to stop the development server, and you can deactivate the virtual environment with the exit command.

<br/>

```
ctrl+C or exit
```

<br/>

</p>


<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>


<br/>



### *_Project Specifications_*

<p align = "justify">

<b>1. Data and Database</b>
> The box office analytics project, Cinemalytics, sources the box office information from The TMDb (The Movie Database), a comprehensive film database that provides information concerning films, which includes details such as titles, ratings, release dates, revenue, genres, and so on, to be employed for Big Data processing techniques.

<br/>

> The sourced database contains more than 963,000 various films which were released as far as from the year 1899 to 2023. By means of this, those details such as genres, adult, vote count, popularity, and vote average offer a nuanced understanding of audience preferences, film reception, and marketing appeal; poster path helps to visualize the nature and impression within the film; while the budget contributes as the core for the box office status based on 2.5x break–even multiplier and revenue prediction.

<br/>

<b>2. Big Data-Driven Box Office Prediction</b>
> Within the box office analytics project, Cinemalytics, a multiple linear regression model, was integrated into the developed web application that draws upon an extensive array of box office features with an R–squared value of at least 0.60. Accordingly, the web application embraces user-centric design to let box office enthusiasts and industry professionals contribute to such predictive prowess.

<br/>

> Users can seamlessly input data across various attributes – budget, vote count, vote average, popularity, and genres – which fosters an interactive and intuitive experience that transcends the conventional boundaries of revenue prediction in the cinematic realm.

<br/>

<b>3. Data Visualization</b>
> The analytics journey of this box office analytics project, Cinemalytics, extends beyond mere prediction; as such employs a diverse arsenal of visualization techniques. From the nuanced depiction of value versus residual distributions through a bell curve and histogram to the dynamic portrayal of value trends via line graph and scatter plot, the platform transforms raw data into visual narratives.

<br/>

> Accordingly, the insights are further enriched by the integration of a radar chart and correlation matrix, which offers a holistic perspective on the intricate relationships within box office features.

</p>

<br/>


<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>


<br/>

### *_Software License_*

<p align = "justify">

Currently, the box office analytics project, Cinemalytics, is distributed under the MIT License. Hence, look up the <a href = "https://github.com/alennejasper/Cinemalytics/LICENSE.txt"><b>LICENSE.txt</b></a> for more information concerning such.

<br/>

</p>

<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>


<br/>


### *_Acknowledgements_*

<p align = "justify">

First and foremost, box office analytics project, Cinemalytics, would not have been made possible without the efforts and hard work produced by fellow contributors and developers in the group. Lest without these following individuals, the entire project would never have succeeded.

<br/>

Finally, the group would like to take this opportunity to express the deepest gratitude to the subject adviser, Sir Lumer Jude Doce, for his exemplary guidance and feedback throughout the project development. His perspective criticism inspired the group to continue working.

<br/>

</p>

<p align = "right">(<a href = "#readme-ceiling">back to top</a>)</p>