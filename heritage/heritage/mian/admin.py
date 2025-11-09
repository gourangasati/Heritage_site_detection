from django.contrib import admin
from .models import GuideBooking

# Register your models here.

@admin.register(GuideBooking)
class GuideBookingAdmin(admin.ModelAdmin):
    list_display = ('visitor_name', 'site_name', 'visit_date', 'email', 'phone', 'status', 'booking_date')
    list_filter = ('status', 'site_name', 'visit_date', 'booking_date')
    search_fields = ('visitor_name', 'email', 'phone', 'site_name')
    readonly_fields = ('booking_date',)
    date_hierarchy = 'visit_date'
