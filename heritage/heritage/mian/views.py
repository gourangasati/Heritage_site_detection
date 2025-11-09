# mian/views.py
import os
import json
import numpy as np
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .forms import ImageUploadForm
from django.conf import settings
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model on server start (global)
MODEL_PATH = os.path.join(settings.BASE_DIR, '..', 'model', 'heritage_model.h5')
CLASS_PATH = os.path.join(settings.BASE_DIR, '..', 'model', 'class_indices.json')

model = None
class_indices = None
idx_to_label = None

def get_site_info(site_name):
    """Get detailed information about a heritage site"""
    sites_info = {
        'Taj Mahal': {
            'name': 'Taj Mahal',
            'location': 'Agra, Uttar Pradesh',
            'description': 'A white marble mausoleum built by Mughal emperor Shah Jahan',
            'built_year': '1632-1653',
            'architect': 'Ustad Ahmad Lahauri',
            'significance': 'UNESCO World Heritage Site, one of the Seven Wonders of the World',
            'visiting_hours': '6:00 AM - 7:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹1100 for foreigners',
            'history': 'Built in memory of Mumtaz Mahal, the favorite wife of Shah Jahan. It took 22 years and 20,000 workers to complete.',
            'facts': [
                'Made entirely of white marble',
                'Changes color throughout the day',
                'Took 22 years to complete',
                'Uses Islamic, Persian, and Indian architectural styles'
            ]
        },
        'Qutub Minar': {
            'name': 'Qutub Minar',
            'location': 'Delhi',
            'description': 'A 73-meter tall minaret, a UNESCO World Heritage Site',
            'built_year': '1193',
            'architect': 'Qutb-ud-din Aibak',
            'significance': 'UNESCO World Heritage Site, tallest brick minaret in the world',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Built by Qutb-ud-din Aibak, the founder of the Delhi Sultanate. The construction was completed by his successor Iltutmish.',
            'facts': [
                'Tallest brick minaret in the world',
                'Has 379 steps to the top',
                'Survived several earthquakes',
                'Features intricate carvings and inscriptions'
            ]
        },
        'tajmahal': {
            'name': 'Taj Mahal',
            'location': 'Agra, Uttar Pradesh',
            'description': 'A white marble mausoleum built by Mughal emperor Shah Jahan',
            'built_year': '1632-1653',
            'architect': 'Ustad Ahmad Lahauri',
            'significance': 'UNESCO World Heritage Site, one of the Seven Wonders of the World',
            'visiting_hours': '6:00 AM - 7:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹1100 for foreigners',
            'history': 'Built in memory of Mumtaz Mahal, the favorite wife of Shah Jahan. It took 22 years and 20,000 workers to complete.',
            'facts': [
                'Made entirely of white marble',
                'Changes color throughout the day',
                'Took 22 years to complete',
                'Uses Islamic, Persian, and Indian architectural styles'
            ]
        },
        'qutub_minar': {
            'name': 'Qutub Minar',
            'location': 'Delhi',
            'description': 'A 73-meter tall minaret, a UNESCO World Heritage Site',
            'built_year': '1193',
            'architect': 'Qutb-ud-din Aibak',
            'significance': 'UNESCO World Heritage Site, tallest brick minaret in the world',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Built by Qutb-ud-din Aibak, the founder of the Delhi Sultanate. The construction was completed by his successor Iltutmish.',
            'facts': [
                'Tallest brick minaret in the world',
                'Has 379 steps to the top',
                'Survived several earthquakes',
                'Features intricate carvings and inscriptions'
            ]
        },
        'gateway_of_india': {
            'name': 'Gateway of India',
            'location': 'Mumbai, Maharashtra',
            'description': 'A monument built to commemorate the visit of King George V',
            'built_year': '1911-1924',
            'architect': 'George Wittet',
            'significance': 'Iconic monument and major tourist attraction in Mumbai',
            'visiting_hours': '24 hours (open all day)',
            'best_time': 'November to February',
            'entry_fee': 'Free',
            'history': 'Built to commemorate the landing of King George V and Queen Mary in Mumbai in 1911. It was completed in 1924.',
            'facts': [
                'Built in Indo-Saracenic style',
                'Made of yellow basalt and reinforced concrete',
                'Last British troops left India through this gate',
                'Major tourist attraction in Mumbai'
            ]
        }
    }
    
    # Normalize site name
    site_name_normalized = site_name.replace(' ', '_').lower()
    if site_name_normalized in sites_info:
        return sites_info[site_name_normalized]
    
    # Try direct match
    if site_name in sites_info:
        return sites_info[site_name]
    
    # Try case-insensitive match
    for key, value in sites_info.items():
        if key.lower() == site_name.lower():
            return value
    
    return None

def load_resources():
    global model, class_indices, idx_to_label
    try:
        if model is None and os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            if not os.path.exists(MODEL_PATH):
                print(f"Warning: Model file not found at {MODEL_PATH}")
        
        if class_indices is None and os.path.exists(CLASS_PATH):
            print(f"Loading class indices from: {CLASS_PATH}")
            with open(CLASS_PATH, 'r') as f:
                class_indices = json.load(f)
            idx_to_label = {v: k for k, v in class_indices.items()}
            print(f"Class indices loaded: {class_indices}")
        else:
            if not os.path.exists(CLASS_PATH):
                print(f"Warning: Class indices file not found at {CLASS_PATH}")
    except Exception as e:
        print(f"Error: Could not load model resources: {e}")
        import traceback
        traceback.print_exc()
        model = None
        class_indices = None
        idx_to_label = None

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def index(request):
    form = ImageUploadForm()
    # Try to load model when index page is accessed
    load_resources()
    return render(request, 'index.html', {'form': form})

@csrf_exempt
def predict_view(request):
    if request.method == 'POST':
        try:
            form = ImageUploadForm(request.POST, request.FILES)
            if form.is_valid():
                # Load resources if not already loaded
                load_resources()
                
                if model is None or idx_to_label is None:
                    error_msg = f'Model not found. Model path: {MODEL_PATH}, Exists: {os.path.exists(MODEL_PATH)}'
                    print(f"Error: {error_msg}")
                    return JsonResponse({'error': error_msg}, status=500)
                
                img = form.cleaned_data['image']
                
                # Ensure media directory exists
                os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
                
                save_path = os.path.join(settings.MEDIA_ROOT, img.name)
                with open(save_path, 'wb+') as dest:
                    for chunk in img.chunks():
                        dest.write(chunk)
                
                # Preprocess and predict
                x = preprocess_image(save_path)
                preds = model.predict(x, verbose=0)[0]
                top_idx = preds.argmax()
                confidence = float(preds[top_idx])
                label = idx_to_label[top_idx]
                
                # Get site information
                site_info = get_site_info(label)
                
                data = {
                    'label': label,
                    'confidence': round(confidence, 4),
                    'all': {idx_to_label[i]: float(preds[i]) for i in range(len(preds))},
                    'site_info': site_info
                }
                
                # Always return JSON for API clients
                # Check if it's an AJAX request by checking headers
                if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.content_type == 'application/json':
                    return JsonResponse(data)
                # For regular form submissions, also return JSON (since we're using fetch)
                return JsonResponse(data)
            else:
                return JsonResponse({'error': 'Invalid form data. Please upload a valid image file.'}, status=400)
        except Exception as e:
            error_msg = f'Prediction error: {str(e)}'
            print(f"Exception in predict_view: {error_msg}")
            import traceback
            traceback.print_exc()
            return JsonResponse({'error': error_msg}, status=500)
    return JsonResponse({'error': 'invalid request'}, status=400)

def search(request):
    """Search for heritage sites"""
    # List of available heritage sites with detailed information
    all_sites = [
        {
            'name': 'Taj Mahal',
            'location': 'Agra, Uttar Pradesh',
            'description': 'A white marble mausoleum built by Mughal emperor Shah Jahan',
            'built_year': '1632-1653',
            'architect': 'Ustad Ahmad Lahauri',
            'significance': 'UNESCO World Heritage Site, one of the Seven Wonders of the World',
            'visiting_hours': '6:00 AM - 7:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹1100 for foreigners',
            'history': 'Built in memory of Mumtaz Mahal, the favorite wife of Shah Jahan. It took 22 years and 20,000 workers to complete.'
        },
        {
            'name': 'Qutub Minar',
            'location': 'Delhi',
            'description': 'A 73-meter tall minaret, a UNESCO World Heritage Site',
            'built_year': '1193',
            'architect': 'Qutb-ud-din Aibak',
            'significance': 'UNESCO World Heritage Site, tallest brick minaret in the world',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Built by Qutb-ud-din Aibak, the founder of the Delhi Sultanate. The construction was completed by his successor Iltutmish.'
        },
        {
            'name': 'Gateway of India',
            'location': 'Mumbai, Maharashtra',
            'description': 'A monument built to commemorate the visit of King George V',
            'built_year': '1911-1924',
            'architect': 'George Wittet',
            'significance': 'Iconic monument and major tourist attraction in Mumbai',
            'visiting_hours': '24 hours (open all day)',
            'best_time': 'November to February',
            'entry_fee': 'Free',
            'history': 'Built to commemorate the landing of King George V and Queen Mary in Mumbai in 1911. It was completed in 1924.'
        },
    ]
    
    query = request.GET.get('q', '')
    if query:
        query_lower = query.lower()
        heritage_sites = [site for site in all_sites if query_lower in site['name'].lower() or query_lower in site['location'].lower()]
    else:
        heritage_sites = all_sites
    
    return render(request, 'search.html', {'heritage_sites': heritage_sites, 'query': query})

def guide(request):
    """Guide booking page"""
    from .models import GuideBooking
    from .forms import GuideBookingForm
    
    heritage_sites = [
        {'name': 'Taj Mahal', 'location': 'Agra, Uttar Pradesh'},
        {'name': 'Qutub Minar', 'location': 'Delhi'},
        {'name': 'Gateway of India', 'location': 'Mumbai, Maharashtra'},
    ]
    
    if request.method == 'POST':
        # Handle guide booking
        form = GuideBookingForm(request.POST)
        if form.is_valid():
            try:
                # Save booking to database
                booking = GuideBooking.objects.create(
                    site_name=form.cleaned_data['site_name'],
                    visitor_name=form.cleaned_data['visitor_name'],
                    email=form.cleaned_data['email'],
                    phone=form.cleaned_data['phone'],
                    visit_date=form.cleaned_data['visit_date'],
                    group_size=form.cleaned_data['group_size'],
                    language=form.cleaned_data['language'],
                    special_requirements=form.cleaned_data.get('special_requirements', ''),
                    status='pending'
                )
                
                print(f"Booking saved: {booking}")
                
                return render(request, 'guide.html', {
                    'heritage_sites': heritage_sites,
                    'form': GuideBookingForm(),  # Reset form after successful submission
                    'booking_success': True,
                    'booking_details': {
                        'site_name': form.cleaned_data['site_name'],
                        'visitor_name': form.cleaned_data['visitor_name'],
                        'visit_date': form.cleaned_data['visit_date'],
                        'booking_id': booking.id
                    }
                })
            except Exception as e:
                print(f"Error saving booking: {e}")
                import traceback
                traceback.print_exc()
                return render(request, 'guide.html', {
                    'heritage_sites': heritage_sites,
                    'form': form,
                    'booking_error': f'Error saving booking: {str(e)}'
                })
        else:
            # Form is invalid, return with errors
            return render(request, 'guide.html', {
                'heritage_sites': heritage_sites,
                'form': form,
                'form_errors': form.errors
            })
    
    # GET request - show empty form
    form = GuideBookingForm()
    return render(request, 'guide.html', {
        'heritage_sites': heritage_sites,
        'form': form
    })

def about(request):
    """About page"""
    return render(request, 'about.html')

def profile(request):
    """User profile page"""
    # Mock user data - in real app, this would come from database
    user_data = {
        'name': 'Heritage Explorer',
        'email': 'user@heritagedetector.com',
        'total_detections': 42,
        'favorite_sites': ['Taj Mahal', 'Qutub Minar', 'Gateway of India'],
        'recent_detections': [
            {'site': 'Taj Mahal', 'date': '2025-11-09', 'confidence': '99.5%'},
            {'site': 'Qutub Minar', 'date': '2025-11-08', 'confidence': '98.2%'},
            {'site': 'Gateway of India', 'date': '2025-11-07', 'confidence': '97.8%'},
        ]
    }
    return render(request, 'profile.html', {'user': user_data})
