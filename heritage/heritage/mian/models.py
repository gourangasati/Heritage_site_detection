from django.db import models
from django.utils import timezone

# Create your models here.

class GuideBooking(models.Model):
    """Model to store guide booking information"""
    site_name = models.CharField(max_length=200)
    visitor_name = models.CharField(max_length=200)
    email = models.EmailField()
    phone = models.CharField(max_length=20)
    visit_date = models.DateField()
    group_size = models.CharField(max_length=50)
    language = models.CharField(max_length=50)
    special_requirements = models.TextField(blank=True, null=True)
    booking_date = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, default='pending', choices=[
        ('pending', 'Pending'),
        ('confirmed', 'Confirmed'),
        ('cancelled', 'Cancelled'),
    ])
    
    class Meta:
        ordering = ['-booking_date']
        verbose_name = 'Guide Booking'
        verbose_name_plural = 'Guide Bookings'
    
    def __str__(self):
        return f"{self.visitor_name} - {self.site_name} - {self.visit_date}"
