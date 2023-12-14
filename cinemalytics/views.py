from django.shortcuts import render
from django.db.models import Sum, Min, Max
from .forms import DescriptionForm, PredictionForm
from io import BytesIO
from scipy.stats import norm, skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from .models import *

import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plot
import seaborn as seaborn
import base64

def line_graph(queryset, start_date, end_date, numeric_attributes, form):
    date_range = pandas.date_range(start = start_date, end = end_date, freq = "Y")

    verbose = {
        "title": "Title",
        "vote_average": "Vote Average",
        "vote_count": "Vote Count",
        "release_date": "Release Date",
        "revenue": "Revenue",
        "runtime": "Runtime",
        "adult": "Adult",
        "budget": "Budget",
        "popularity": "Popularity",
        "poster_path": "Poster Path",
        "genres": "Genres",
        "mean": "Mean", 
        "median": "Median", 
        "mode": "Mode", 
        "standard_error": "Standard Error of the Mean", 
        "standard_deviation": "Standard Deviation", 
        "value_present": "N", 
        "value_missing": "Missing", 
        "sum": "Sum", 
        "minimum": "Minimum", 
        "maximum": "Maximum", 
        "skewness": "Skewness", 
        "tailedness": "Kurtosis", 
        "variance": "Variance"
    }

    metrics = [metric for metric in ["mean", "median", "mode", "standard_error", "standard_deviation", "value_present", "value_missing", "sum", "minimum", "maximum", "skewness", "tailedness", "variance"] if form.cleaned_data[metric]]

    line_charts = []

    for metric in metrics:
        values_dictionary = {}

        for field in numeric_attributes:
            values_list = []

            for year in date_range:
                filtered_queryset = queryset.filter(release_date__year = year.year)

                values = filtered_queryset.values_list(field, flat = True)

                statistics = {}
                if "mean" in metrics:
                    statistics['mean'] = pandas.Series(values).mean()

                if "median" in metrics:
                    statistics["median"] = pandas.Series(values).median()

                if "mode" in metrics:
                    mode_value = pandas.Series(values).mode()
                    statistics["mode"] = mode_value.iloc[0] if not mode_value.empty else None

                if "standard_error" in metrics:
                    statistics["standard_error"] = pandas.Series(values).sem()

                if "standard_deviation" in metrics:
                    statistics["standard_deviation"] = pandas.Series(values).std()

                if "sum" in metrics:
                    statistics["sum"] = pandas.Series(values).sum()

                if "minimum" in metrics:
                    statistics["minimum"] = pandas.Series(values).min()

                if "maximum" in metrics:
                    statistics["maximum"] = pandas.Series(values).max()

                if "variance" in metrics:
                    statistics["variance"] = pandas.Series(values).var()

                present_count = filtered_queryset.count()
                if "value_present" in metrics:
                    statistics["value_present"] = present_count

                if "value_missing" in metrics:
                    missing_count = present_count - filtered_queryset.exclude(**{field: None}).count()

                    statistics['value_missing'] = missing_count
                
                if "skewness" in metrics:
                    statistics["skewness"] = skew(values)
                
                if "tailedness" in metrics:
                    statistics["tailedness"] = kurtosis(values)

                value = statistics.get(metric)
                if value is not None:
                    values_list.append(value)

            values_dictionary[field] = values_list

        with BytesIO() as buffer:
            figure, axis = plot.subplots(figsize = (30, 20))

            figure.set_facecolor("#6E6A6A")

            axis.set_facecolor("#6E6A6A")

            colormap = plot.cm.get_cmap("Wistia", len(numeric_attributes))

            for iteration, (field, values_list) in enumerate(values_dictionary.items()):
                axis.plot(date_range, values_list, label = f"{verbose[field]} {verbose[metric]}", marker = "o", zorder = 3, color = colormap(iteration / len(numeric_attributes)))

            font_properties = {"fontfamily": "Montserrat", "weight": "normal", "fontsize": 26}

            axis.set_xlabel("Date Range", **font_properties, color = "#FFEBA7")
            
            axis.set_ylabel(f"{verbose[metric]} Values", **font_properties, color = "#FFEBA7")

            axis.yaxis.set_major_formatter(plot.FuncFormatter(lambda y, _: "{:,.2f}".format(y).replace(",", ", ")))

            axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 1, rotation = 20)

            axis.tick_params(axis = "y", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 2, rotation = 20)

            axis.legend(facecolor = "#462113", labelcolor = "#FFEBA7", edgecolor = "#FFEBA7", fontsize = 26, prop = {"family": "Montserrat", "size": 26}, borderpad = 2).get_frame().set_alpha(1)

            axis.grid(True, color = "#462113", linestyle = "-", linewidth = 2, zorder = 0)

            for spine in axis.spines.values():
                spine.set_edgecolor("#462113")

            for spine in axis.spines.values():
                spine.set_linewidth(2)

            figure.savefig(buffer, format = "png")

            buffer.seek(0)

            image = base64.b64encode(buffer.getvalue()).decode("utf-8")

            information = {"metric": verbose[metric], "image": image}

            line_charts.append(information)

    return line_charts


def coefficient_matrix(numeric_attributes, queryset):
    verbose = {
        "title": "Title",
        "vote_average": "Vote Average",
        "vote_count": "Vote Count",
        "release_date": "Release Date",
        "revenue": "Revenue",
        "runtime": "Runtime",
        "adult": "Adult",
        "budget": "Budget",
        "popularity": "Popularity",
        "poster_path": "Poster Path",
        "genres": "Genres",
    }

    values = {verbose[field]: list(queryset.values_list(field, flat = True)) for field in numeric_attributes}

    dataframe = pandas.DataFrame(values)
    
    correlation_matrix = dataframe.corr()
    
    return correlation_matrix

def correlation_heatmap(matrix):
    figure, axis = plot.subplots(figsize = (30, 20))

    figure.set_facecolor("#6E6A6A")

    axis.set_facecolor("#6E6A6A")

    seaborn.heatmap(matrix, ax = axis, annot = True, cmap = "copper", fmt = ".2f", annot_kws = {"family": "Montserrat", "weight": "normal", "size": 26, "color": "#462113"}, linewidths = 2, linecolor = "#462113", clip_on = False, zorder = 3)

    axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 1, rotation = 20)

    axis.tick_params(axis = "y", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 2, rotation = 20)

    axis.grid(True, color = "#462113", linestyle = "-", linewidth = 2, zorder = 0)

    cbar = axis.collections[0].colorbar
        
    cbar.ax.yaxis.set_tick_params(color = "#FFEBA7", width = 2, zorder = 4)

    cbar.ax.yaxis.set_major_formatter(plot.FuncFormatter(lambda y, _: "{:,.2f}".format(y).replace(",", ", ")))

    cbar.outline.set_linewidth(2)
    
    cbar.outline.set_edgecolor("#462113")

    cbar.ax.tick_params(axis = "y", color = "#462113", width = 2, labelsize = 26, labelfontfamily = "Montserrat", rotation = 20)
    for label in cbar.ax.yaxis.get_ticklabels():
        label.set_color("#FFEBA7")

    for spine in axis.spines.values():
        spine.set_edgecolor("#462113")

    for spine in axis.spines.values():
        spine.set_linewidth(2)
        
    buffer = BytesIO()
    
    plot.savefig(buffer, format = "png")
    
    buffer.seek(0)
    
    plot.close()

    image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return image

def probability_distribution(values):
    mean = numpy.mean(values)

    standard_deviation = numpy.std(values)

    dist = norm(loc = mean, scale = standard_deviation)

    x = numpy.linspace(min(values), max(values), 100)

    y = dist.pdf(x)

    return x.tolist(), y.tolist()

def distribution_map(values, field):
    verbose = {
        "title": "Title",
        "vote_average": "Vote Average",
        "vote_count": "Vote Count",
        "release_date": "Release Date",
        "revenue": "Revenue",
        "runtime": "Runtime",
        "adult": "Adult",
        "budget": "Budget",
        "popularity": "Popularity",
        "poster_path": "Poster Path",
        "genres": "Genres",
    }

    figure, axis = plot.subplots(figsize = (30, 20))

    figure.set_facecolor("#6E6A6A")

    axis.set_facecolor("#6E6A6A")

    seaborn.kdeplot(values, color = "#FFEBA7", fill = True, common_norm = False, zorder = 3)

    mean, standard_deviation = norm.fit(values)

    x_minimum, x_maximum = plot.xlim()

    x = numpy.linspace(x_minimum, x_maximum, 100)

    p = norm.pdf(x, mean, standard_deviation)

    axis.plot(x, p, "k", linewidth = 2, color = "#FFEBA7", zorder = 4)

    axis.axvline(x = mean, color = "#FFEBA7", linestyle = "-", linewidth = 2, label = "Mode", zorder = 5)

    font_properties = {"fontfamily": "Montserrat", "weight": "normal", "fontsize": 26}

    axis.set_xlabel(f"{verbose[field]} Values", **font_properties, color = "#FFEBA7")

    axis.set_ylabel(f"{verbose[field]} Density", **font_properties, color = "#FFEBA7")

    axis.xaxis.set_major_formatter(plot.FuncFormatter(lambda x, _: "{:,.2f}".format(x).replace(",", ", ")))

    axis.yaxis.set_major_formatter(plot.FuncFormatter(lambda y, _: "{:,.2f}".format(y).replace(",", ", ")))

    axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 1, rotation = 20)

    axis.tick_params(axis = "y", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 2, rotation = 20)

    axis.legend(["Kernel Density", "Fitted Distribution"], facecolor = "#462113", labelcolor = "#FFEBA7", edgecolor = "#FFEBA7", fontsize = 26, prop = {"family": "Montserrat", "size": 26}, borderpad = 2).get_frame().set_alpha(1)

    axis.grid(True, color = "#462113", linestyle = "-", linewidth = 2, zorder = 0)

    for spine in axis.spines.values():
        spine.set_edgecolor("#462113")

    for spine in axis.spines.values():
        spine.set_linewidth(2)

    for spine in axis.spines.values():
        spine.set_zorder(1)

    buffer = BytesIO()

    plot.savefig(buffer, format = "png")

    buffer.seek(0)

    plot.close("all")

    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image

def describe(request):
    if request.method == "POST":
        form = DescriptionForm(request.POST)

        if form.is_valid():
            numeric_attributes = form.cleaned_data["numeric_attributes"]

            nonnumeric_attributes = form.cleaned_data["nonnumeric_attributes"]

            start_date = form.cleaned_data["start_date"]
    
            end_date = form.cleaned_data["end_date"]
    
            mean = form.cleaned_data["mean"]
    
            mode = form.cleaned_data["mode"]
    
            median = form.cleaned_data["median"]
    
            probability_distribution = form.cleaned_data["probability_distribution"]
    
            correlation_matrix = form.cleaned_data["correlation_matrix"]
    
            standard_deviation = form.cleaned_data["standard_deviation"]
    
            standard_error = form.cleaned_data["standard_error"]
    
            value_present = form.cleaned_data["value_present"]
    
            value_missing = form.cleaned_data["value_missing"]
    
            line_chart = form.cleaned_data["line_chart"]

            sum = form.cleaned_data["sum"]

            minimum = form.cleaned_data["minimum"]

            maximum = form.cleaned_data["maximum"]

            skewness = form.cleaned_data["skewness"]
            
            tailedness = form.cleaned_data["tailedness"]

            variance = form.cleaned_data["variance"]

            queryset = Film.objects.all()

            if start_date and end_date:
                queryset = queryset.filter(release_date__range = (start_date, end_date))
            
            verbose = {
                "title": "Title",
                "vote_average": "Vote Average",
                "vote_count": "Vote Count",
                "release_date": "Release Date",
                "revenue": "Revenue",
                "runtime": "Runtime",
                "adult": "Adult",
                "budget": "Budget",
                "popularity": "Popularity",
                "poster_path": "Poster Path",
                "genres": "Genres",
            }

            mean_values = {}
            if mean:
                mean_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).mean()).replace(",", ", ") if mean else None for field in numeric_attributes}

            mode_values = {}
            if mode:
                mode_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).mode().iloc[0]).replace(",", ", ") if mode else None for field in numeric_attributes}
        
            median_values = {}
            if median:
                median_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).median()).replace(",", ", ") if median else None for field in numeric_attributes}

            deviation_values = {}
            if standard_deviation:
                deviation_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).std()).replace(",", ", ") if standard_deviation else None for field in numeric_attributes}

            distribution_graph = {}
            if probability_distribution:
                distribution_graph = {verbose[field]: distribution_map(queryset.values_list(field, flat = True), field) if probability_distribution else None for field in numeric_attributes}

            error_values = {}
            if standard_error:
                error_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).sem()).replace(",", ", ") if standard_error else None for field in numeric_attributes}

            present_values = {}
            if value_present:
                present_values = {verbose[field]: "{:,.2f}".format(len(queryset.values_list(field, flat = True))).replace(",", ", ") if value_present else None for field in set(numeric_attributes) | set(nonnumeric_attributes)}

            missing_values = {}
            if value_missing:
                missing_values = {verbose[field]: "{:,.2f}".format(queryset.filter(**{field: None}).count()).replace(",", ", ") if value_missing else None for field in set(numeric_attributes) | set(nonnumeric_attributes)}
            
            correlation_matrices = {}
            
            correlation_heatmaps = {}

            field_names = []
            if correlation_matrix:
                correlation_matrices = coefficient_matrix(numeric_attributes, queryset)
                
                correlation_heatmaps = correlation_heatmap(correlation_matrices)

                field_names = correlation_matrices.columns.tolist()

            line_charts = {}
            if line_chart:
                line_charts = line_graph(queryset, start_date, end_date, numeric_attributes, form)
            
            sum_values = {}
            if sum:
                sum_values = {verbose[field]: "{:,.2f}".format(queryset.aggregate(Sum(field))[f"{field}__sum"]).replace(",", ", ") if sum else None for field in numeric_attributes}                
            
            minimum_value = {}
            if minimum:
                minimum_value = {verbose[field]: "{:,.2f}".format(queryset.aggregate(Min(field))[f"{field}__min"]).replace(",", ", ") if minimum else None for field in numeric_attributes}

            maximum_value = {}
            if maximum:
                maximum_value = {verbose[field]: "{:,.2f}".format(queryset.aggregate(Max(field))[f"{field}__max"]).replace(",", ", ") if maximum else None for field in numeric_attributes}
            
            skewness_values = {}
            if skewness:
                skewness_values = {verbose[field]: "{:,.2f}".format(skew(queryset.values_list(field, flat = True))).replace(",", ", ") if skewness else None for field in numeric_attributes}

            kurtosis_values = {}
            if tailedness:
                kurtosis_values = {verbose[field]: "{:,.2f}".format(kurtosis(queryset.values_list(field, flat = True))).replace(",", ", ") if tailedness else None for field in numeric_attributes}
    
            variance_values = {}
            if variance:
                variance_values = {verbose[field]: "{:,.2f}".format(pandas.Series(queryset.values_list(field, flat = True)).var()).replace(",", ", ") if variance else None for field in numeric_attributes}


            context = {"form": form, "mean_values": mean_values, "mode_values": mode_values, "median_values": median_values, "deviation_values": deviation_values, "distribution_graph": distribution_graph, "correlation_heatmaps": correlation_heatmaps, "error_values": error_values, "present_values": present_values, "missing_values": missing_values, "line_charts": line_charts, "sum_values": sum_values, "minimum_value": minimum_value, "maximum_value": maximum_value, "skewness_values": skewness_values, "kurtosis_values": kurtosis_values, "variance_values": variance_values, "field_names": field_names}
            
            return render(request, "outcome.html", context)
    else:
        form = DescriptionForm()

    context = {"form": form, "mean_values": None, "mode_values": None, "median_values": None, "deviation_values": None, "distribution_graph": False, "distribution_data": None, "correlation_heatmaps": None, "error_values": None, "present_values": None, "missing_values": None, "line_charts": None, "sum_values": None, "minimum_value": None, "maximum_value": None, "skewness_values": None, "kurtosis_values": None, "variance_values": None, "field_names": None}
    
    return render(request, "description.html", context)

def radar_chart(attributes_influence):
    verbose = {
        "title": "Title",
        "vote_average": "Vote Average",
        "vote_count": "Vote Count",
        "release_date": "Release Date",
        "revenue": "Revenue",
        "runtime": "Runtime",
        "adult": "Adult",
        "budget": "Budget",
        "popularity": "Popularity",
        "poster_path": "Poster Path",
        "genres": "Genres",
        "Romance": "Romance", 
        "Action": "Action", 
        "Drama": "Drama", 
        "Comedy": "Comedy", 
        "Horror": "Horror", 
        "Thriller": "Thriller", 
        "Crime": "Crime", 
        "Mystery": "Mystery", 
        "Adventure": "Adventure", 
        "Science Fiction": "Science Fiction", 
        "History": "History",
        "Animation": "Animation", 
        "Family": "Family", 
        "Documentary": "Documentary", 
        "Fantasy": "Fantasy", 
        "Music": "Music", 
        "TV Movie": "TV Movie", 
        "War": "War",
        "Western": "Western"
    }

    fields = list(attributes_influence.keys())

    values = numpy.array(list(attributes_influence.values()))

    number_fields = len(fields)

    angles = numpy.linspace(0, 2 * numpy.pi, number_fields, endpoint=False)

    values = numpy.concatenate((values, [values[0]]))

    angles = numpy.concatenate((angles, [angles[0]]))

    figure, axis = plot.subplots(figsize = (30, 20), subplot_kw = dict(polar = True))

    figure.set_facecolor("#6E6A6A")

    axis.set_facecolor("#6E6A6A")

    axis.set_yticklabels([], zorder = 6)

    axis.set_xticks(angles[:-1])

    font_properties = {"fontfamily": "Montserrat", "weight": "normal", "fontsize": 26}

    axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 4)

    axis.yaxis.grid(color = "#462113", linewidth = 2, zorder = 2)

    axis.xaxis.grid(color = "#462113", linewidth = 2, zorder = 1)

    axis.spines["polar"].set_color("#462113")

    axis.spines["polar"].set_linewidth(2)

    axis.spines["polar"].set_zorder(0)

    axis.fill(angles, values, color = "#FFEBA7", alpha = 1, zorder = 3, linewidth = 2)

    angle_labels = [f"{verbose[field]}\n{percentage:.2f}%" for field, percentage in attributes_influence.items()]

    axis.set_xticklabels(angle_labels, **font_properties, zorder = 5)

    buffer = BytesIO()

    plot.savefig(buffer, format = "png")

    plot.close(figure)

    buffer.seek(0)

    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image

def scatter_plot(actual_values, predicted_values):
    figure, axis = plot.subplots(figsize = (30, 20))

    figure.set_facecolor("#6E6A6A")

    axis.set_facecolor("#6E6A6A")

    axis.scatter(actual_values, predicted_values, color = "#FFEBA7", edgecolors = "#FFEBA7", alpha = 1, linewidth = 2, zorder = 3)

    axis.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color = "#FFEBA7", linestyle = "-", linewidth = 2, zorder = 4)

    font_properties = {"fontfamily": "Montserrat", "weight": "normal", "fontsize": 26}

    axis.set_xlabel("Actual Revenue", **font_properties, color = "#FFEBA7")

    axis.set_ylabel("Predicted Revenue", **font_properties, color = "#FFEBA7")

    axis.xaxis.set_major_formatter(plot.FuncFormatter(lambda x, _: "{:,.2f}".format(x).replace(",", ", ")))

    axis.yaxis.set_major_formatter(plot.FuncFormatter(lambda y, _: "{:,.2f}".format(y).replace(",", ", ")))

    axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 1, rotation = 20)

    axis.tick_params(axis = "y", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 2, rotation = 20)
        
    axis.grid(True, color = "#462113", linestyle = "-", linewidth = 2, zorder = 0)

    for spine in axis.spines.values():
        spine.set_edgecolor("#462113")

    for spine in axis.spines.values():
        spine.set_linewidth(2)

    buffer = BytesIO()

    plot.savefig(buffer, format = "png")

    plot.close(figure)

    buffer.seek(0)

    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image


def residual_histogram(actual_values, predicted_values):
    residuals = actual_values - predicted_values

    figure, axis = plot.subplots(figsize = (30, 20))

    figure.set_facecolor("#6E6A6A")

    axis.set_facecolor("#6E6A6A")

    axis.hist(residuals, bins = 20, color = "#FFEBA7", alpha = 1, zorder = 3)
        
    font_properties = {"fontfamily": "Montserrat", "weight": "normal", "fontsize": 26}
        
    axis.tick_params(axis = "x", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 1, rotation = 20)

    axis.tick_params(axis = "y", labelcolor = "#FFEBA7", labelsize = 26, labelfontfamily = "Montserrat", width = 2, zorder = 2, rotation = 20)

    axis.set_xlabel("Residuals Count", **font_properties, color = "#FFEBA7")
    
    axis.set_ylabel("Residuals Frequency", **font_properties, color = "#FFEBA7")

    axis.xaxis.set_major_formatter(plot.FuncFormatter(lambda x, _: "{:,.2f}".format(x).replace(",", ", ")))

    axis.yaxis.set_major_formatter(plot.FuncFormatter(lambda y, _: "{:,.2f}".format(y).replace(",", ", ")))

    axis.grid(True, color = "#462113", linestyle = "-", linewidth = 2, zorder = 0)

    for spine in axis.spines.values():
        spine.set_edgecolor("#462113")

    for spine in axis.spines.values():
        spine.set_linewidth(2)

    buffer = BytesIO()

    plot.savefig(buffer, format = "png")
    
    plot.close(figure)
    
    buffer.seek(0)

    image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image

def predict(request):
    form = PredictionForm(request.POST or None)

    if form.is_valid():
        budget = form.cleaned_data["budget"]

        vote_average = form.cleaned_data["vote_average"]
        
        vote_count = form.cleaned_data["vote_count"]
        
        runtime = form.cleaned_data["runtime"]
        
        genres = form.cleaned_data["genres"]

        popularity = form.cleaned_data["popularity"]

        adult = form.cleaned_data["adult"]

        film_data = Film.objects.values("budget", "vote_count", "vote_average", "runtime", "genres", "revenue", "popularity", "adult", "title")
        
        dataframe = pandas.DataFrame.from_records(film_data)

        dataframe_genres = dataframe["genres"].str.get_dummies(sep = ", ")
        
        dataframe = pandas.concat([dataframe, dataframe_genres], axis = 1)

        dataframe = dataframe.drop("genres", axis = 1)

        user_genres = [True if genre in genres else False for genre in dataframe_genres.columns]

        X = dataframe[["budget", "vote_average", "runtime", "vote_count", "popularity", "adult"] + list(dataframe_genres.columns)]
        
        y = dataframe["revenue"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

        model = LinearRegression()
        
        model.fit(X_train, y_train)

        user_input = [[budget, vote_average, runtime, vote_count, popularity, adult] + user_genres]

        predicted_revenue = model.predict(user_input)

        predicted_status = "Hit" if predicted_revenue[0] >= 2.5 * budget else "Flop"

        coefficients = model.coef_

        overall_influence = sum(abs(coeff * user_input[0][value]) for value, coeff in enumerate(coefficients))
        
        attributes_influence = {field: (abs(coeff * user_input[0][value]) / overall_influence) * 100 for value, (field, coeff) in enumerate(zip(X.columns, coefficients))}

        y_prediction = model.predict(X_test)

        r = r2_score(y_test, y_prediction)
        
        root_mean = numpy.sqrt(mean_squared_error(y_test, y_prediction))

        film_details = pandas.read_csv("datasets/cinemalytics.csv")

        film_details["difference"] = film_details["revenue"].apply(lambda x: abs(x - predicted_revenue[0]))
        
        sort_films = film_details.sort_values(by = "difference")

        dataframe = sort_films.head(4)

        dataframe["poster_url"] = "https://image.tmdb.org/t/p/original/" + dataframe["poster_path"].astype(str)

        dataframe["release_year"] = pandas.to_datetime(dataframe["release_date"]).dt.year

        dataframe["revenue"] = dataframe["revenue"].apply(lambda x: "{:,.2f}".format(x).replace(",", ", "))

        filter_films = dataframe.to_dict(orient = "records")

        radar_graph = radar_chart(attributes_influence)

        scatter_graph = scatter_plot(y_test, y_prediction)

        residual_graph = residual_histogram(y_test, y_prediction)

        context = {"predicted_revenue": "{:,.2f}".format(predicted_revenue[0]).replace(",", ", "), "r": "{:,.2f}".format(r).replace(",", ", "), "root_mean": "{:,.2f}".format(root_mean).replace(",", ", "), "predicted_status": predicted_status, "attributes_influence": attributes_influence, "filter_films": filter_films, "radar_graph": radar_graph, "scatter_graph": scatter_graph, "residual_graph": residual_graph}

        return render(request, "outcome.html", context)

    context = {"form": form}

    return render(request, "prediction.html", context)

def enter(request):

    context = {}

    return render(request, "portal.html", context)