from django import forms

class DescriptionForm(forms.Form):
    numeric_choices = [("budget", "Budget"), ("revenue", "Revenue"), ("popularity", "Popularity"), ("vote_average", "Vote Average"), ("vote_count", "Vote Count"), ("runtime", "Runtime")]
    numeric_attributes = forms.MultipleChoiceField(required = False, choices = numeric_choices, widget = forms.CheckboxSelectMultiple)
    nonnumeric_choices = [("genres", "Genres"), ("adult", "Adult"), ("release_date", "Release Date"), ("poster_path", "Poster Path")]
    nonnumeric_attributes = forms.MultipleChoiceField(required = False, choices = nonnumeric_choices, widget = forms.CheckboxSelectMultiple)
    mean = forms.BooleanField(required = False, initial = False)
    mode = forms.BooleanField(required = False, initial = False)
    median = forms.BooleanField(required = False, initial = False)
    probability_distribution = forms.BooleanField(required = False, initial = False)
    correlation_matrix = forms.BooleanField(required = False, initial = False)
    standard_deviation = forms.BooleanField(required = False, initial = False)
    standard_error = forms.BooleanField(required = False, initial = False)
    value_present = forms.BooleanField(required = False, initial = False)
    value_missing = forms.BooleanField(required = False, initial = False)
    minimum = forms.BooleanField(required = False, initial = False)
    maximum = forms.BooleanField(required = False, initial = False)
    line_chart = forms.BooleanField(required = False, initial = False)
    sum = forms.BooleanField(required = False, initial = False)
    skewness = forms.BooleanField(required = False, initial = False)
    tailedness = forms.BooleanField(required = False, initial = False)
    variance = forms.BooleanField(required = False, initial = False)
    start_date = forms.DateField(required = False, widget = forms.DateInput(attrs = {"type": "date"}))
    end_date = forms.DateField(required = False, widget = forms.DateInput(attrs = {"type": "date"}))

class PredictionForm(forms.Form):
    budget = forms.IntegerField(required = False)
    vote_average = forms.FloatField(required = False)
    popularity = forms.FloatField(required = False)
    runtime = forms.IntegerField(required = False)
    vote_count = forms.IntegerField(required = False)
    adult = forms.BooleanField(required = False, initial = False)
    genre_choices = [("History", "History"), ("Romance", "Romance"), ("Action", "Action"), ("Drama", "Drama"), ("Comedy", "Comedy"), ("Horror", "Horror"), ("Thriller", "Thriller"), ("Crime", "Crime"), ("Mystery", "Mystery"), ("Adventure", "Adventure"), ("Science Fiction", "Science Fiction"), ("Animation", "Animation"), ("Family", "Family"), ("Documentary", "Documentary"), ("Fantasy", "Fantasy"), ("Music", "Music"), ("TV Movie", "TV Movie"), ("War", "War"), ("Western", "Western")]
    genres = forms.MultipleChoiceField(required = False, choices = genre_choices, widget = forms.CheckboxSelectMultiple)