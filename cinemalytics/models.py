from django.db import models
from django.utils.translation import gettext

# Create your models here.
class Film(models.Model):
    title = models.CharField(gettext("title"), max_length = 500)
    
    vote_average = models.FloatField(gettext("vote_average"))
    
    vote_count = models.IntegerField(gettext("vote_count"))
        
    release_date = models.DateField(gettext("release_date"))
    
    revenue = models.IntegerField(gettext("revenue"))
    
    runtime = models.IntegerField(gettext("runtime"))
    
    adult = models.BooleanField(gettext("adult"))
    
    budget = models.IntegerField(gettext("budget"))
        
    popularity = models.FloatField(gettext("popularity"))
    
    poster_path = models.CharField(gettext("poster_path"), max_length = 1500)
    
    genres = models.CharField(gettext("genres"), max_length = 1500)
    
    def poster_url(self):
        base_url = "https://image.tmdb.org/t/p/original/"
        
        return f"{base_url}{self.poster_path}"