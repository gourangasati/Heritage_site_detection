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
MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'heritage_model.h5')
CLASS_PATH = os.path.join(settings.BASE_DIR, 'model', 'class_indices.json')
model = None
class_indices = None
idx_to_label = None

def get_site_info(site_name):
    """Get detailed information about a heritage site"""

    sites_info = {
        'tajmahal': {
            'name': 'Taj Mahal',
            'location': 'Agra, Uttar Pradesh',
            'description': 'A white marble mausoleum built by Mughal emperor Shah Jahan.',
            'built_year': '1632-1653',
            'architect': 'Ustad Ahmad Lahauri',
            'significance': 'UNESCO World Heritage Site, one of the Seven Wonders of the World.',
            'visiting_hours': '6:00 AM - 7:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹1100 for foreigners',
            'history': 'Built in memory of Mumtaz Mahal. Took 22 years and 20,000 workers.',
            'facts': [
                'Made entirely of white marble.',
                'Changes color with daylight.',
                'Symbol of eternal love.'
            ]
        },

        'qutub_minar': {
            'name': 'Qutub Minar',
            'location': 'Delhi',
            'description': 'A 73-meter tall UNESCO World Heritage minaret.',
            'built_year': '1193',
            'architect': 'Qutb-ud-din Aibak',
            'significance': 'Tallest brick minaret in the world.',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Started by Qutb-ud-din Aibak; completed by Iltutmish.',
            'facts': [
                'Has 379 steps.',
                'Built with red sandstone.',
                'Contains intricate carvings.'
            ]
        },
        'India Gate': {
    'name': 'India Gate',
    'location': 'New Delhi',
    'description': 'A war memorial dedicated to Indian soldiers who died during World War I.',
    'built_year': '1921–1931',
    'architect': 'Sir Edwin Lutyens',
    'significance': 'Honors 84,000 soldiers of the British Indian Army; major national monument.',
    'visiting_hours': 'Open 24 hours',
    'best_time': 'October to March',
    'entry_fee': 'Free',
    'history': 'Originally called the All India War Memorial; inaugurated in 1931 by Viceroy Lord Irwin.',
    'facts': [
        'Stands 42 meters tall.',
        'The eternal flame Amar Jawan Jyoti was added in 1971.'
    ]
},


        'Gateway of India': {
            'name': 'Gateway of India',
            'location': 'Mumbai, Maharashtra',
            'description': 'A basalt arch monument built to honor King George V.',
            'built_year': '1911-1924',
            'architect': 'George Wittet',
            'significance': 'Major tourist attraction and historic landmark.',
            'visiting_hours': 'Open 24 hours',
            'best_time': 'November to February',
            'entry_fee': 'Free',
            'history': 'Constructed for the visit of King George V and Queen Mary.',
            'facts': [
                'Indo-Saracenic architectural style.',
                'Overlooks the Arabian Sea.'
            ]
        },

        'Ajanta Caves': {
            'name': 'Ajanta Caves',
            'location': 'Aurangabad, Maharashtra',
            'description': 'Rock-cut Buddhist cave monuments famous for murals.',
            'built_year': '2nd century BCE - 480 CE',
            'architect': 'Ancient Buddhist monks',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '9:00 AM - 5:00 PM',
            'best_time': 'November to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Caves rediscovered in 1819 by British soldiers.',
            'facts': [
                'Contains ancient paintings and sculptures.',
                'Carved entirely in rock.'
            ]
        },

        'Alai Darwaza': {
            'name': 'Alai Darwaza',
            'location': 'Delhi',
            'description': 'A gateway built by Alauddin Khalji.',
            'built_year': '1311',
            'architect': 'Alauddin Khalji',
            'significance': 'Earliest example of Indo-Islamic architecture.',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Part of the Qutub Minar complex.',
            'facts': [
                'Made of red sandstone.',
                'Has intricate latticed stone screens.'
            ]
        },

        'Alai Minar': {
            'name': 'Alai Minar',
            'location': 'Delhi',
            'description': 'Unfinished tower intended to be twice the height of Qutub Minar.',
            'built_year': '1300s',
            'architect': 'Alauddin Khalji',
            'significance': 'Symbol of Khalji empire ambition.',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Construction stopped after Alauddin Khalji died.',
            'facts': [
                'Only the 27m base was completed.',
                'Located inside Qutub Complex.'
            ]
        },

        'Basilica of Bom Jesus': {
            'name': 'Basilica of Bom Jesus',
            'location': 'Old Goa, Goa',
            'description': 'Famous church holding the remains of St. Francis Xavier.',
            'built_year': '1594-1605',
            'architect': 'Domingos Fernandes',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '9:00 AM - 6:30 PM',
            'best_time': 'November to February',
            'entry_fee': 'Free',
            'history': 'Important pilgrimage site for Christians.',
            'facts': [
                'Baroque architecture.',
                'Contains the relics of St. Francis Xavier.'
            ]
        },

        'Charar-E-Sharif': {
            'name': 'Charar-E-Sharif',
            'location': 'Budgam, Jammu & Kashmir',
            'description': 'Holy shrine of Sufi saint Sheikh Noor-ud-Din.',
            'built_year': '15th century',
            'architect': 'Kashmiri artisans',
            'significance': 'Major Sufi pilgrimage site.',
            'visiting_hours': '6:00 AM - 8:00 PM',
            'best_time': 'April to October',
            'entry_fee': 'Free',
            'history': 'A revered religious site rebuilt after fires.',
            'facts': [
                'Center of Kashmiri Sufi culture.',
                'Known for spiritual gatherings.'
            ]
        },

        'Charminar': {
            'name': 'Charminar',
            'location': 'Hyderabad, Telangana',
            'description': 'Iconic monument with four minarets.',
            'built_year': '1591',
            'architect': 'Mir Momin Astarabadi',
            'significance': 'Symbol of Hyderabad.',
            'visiting_hours': '9:00 AM - 5:30 PM',
            'best_time': 'October to February',
            'entry_fee': '₹25 for Indians, ₹300 for foreigners',
            'history': 'Built by Mohammed Quli Qutb Shah.',
            'facts': [
                'Over 400 years old.',
                'Surrounded by Laad Bazaar.'
            ]
        },

        'Chhota Imambara': {
            'name': 'Chhota Imambara',
            'location': 'Lucknow, Uttar Pradesh',
            'description': 'A beautiful Shia monument also known as Palace of Lights.',
            'built_year': '1838',
            'architect': 'Muhammad Ali Shah',
            'significance': 'Important Islamic heritage site.',
            'visiting_hours': '6:00 AM - 5:00 PM',
            'best_time': 'November to February',
            'entry_fee': '₹25 for Indians, ₹300 for foreigners',
            'history': 'Built as a congregation hall for Shia ceremonies.',
            'facts': [
                'Chandeliers imported from Belgium.',
                'Decorated with Arabic calligraphy.'
            ]
        },

        'Ellora Caves': {
            'name': 'Ellora Caves',
            'location': 'Aurangabad, Maharashtra',
            'description': 'Rock-cut caves representing Hindu, Buddhist, and Jain cultures.',
            'built_year': '600-1000 CE',
            'architect': 'Ancient craftsmen',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '6:00 AM - 6:00 PM',
            'best_time': 'November to March',
            'entry_fee': '₹40 for Indians, ₹600 for foreigners',
            'history': 'Includes the world-famous Kailasa Temple.',
            'facts': [
                'Carved completely out of a single rock.',
                '34 major caves in total.'
            ]
        },

        'Fatehpur Sikri': {
            'name': 'Fatehpur Sikri',
            'location': 'Agra, Uttar Pradesh',
            'description': 'Historic Mughal city founded by Akbar.',
            'built_year': '1571',
            'architect': 'Akbar the Great',
            'significance': 'UNESCO World Heritage city.',
            'visiting_hours': '6:00 AM - 6:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹610 for foreigners',
            'history': 'Served as the Mughal capital briefly.',
            'facts': [
                'Known for Buland Darwaza.',
                'Abandoned due to water shortage.'
            ]
        },

        'Golden Temple': {
            'name': 'Golden Temple',
            'location': 'Amritsar, Punjab',
            'description': 'Holistic Sikh Gurudwara covered in gold.',
            'built_year': '1581-1604',
            'architect': 'Guru Arjan Dev Ji',
            'significance': 'Holest shrine of Sikhism.',
            'visiting_hours': '24 hours',
            'best_time': 'November to March',
            'entry_fee': 'Free',
            'history': 'Renovated with gold during Maharaja Ranjit Singh’s rule.',
            'facts': [
                'Serves free food to thousands daily.',
                'Located in the middle of a sacred lake.'
            ]
        },

        'Hawa Mahal': {
            'name': 'Hawa Mahal',
            'location': 'Jaipur, Rajasthan',
            'description': 'Pink sandstone palace known as Palace of Winds.',
            'built_year': '1799',
            'architect': 'Lal Chand Ustad',
            'significance': 'Symbol of Jaipur.',
            'visiting_hours': '9:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹50 for Indians, ₹200 for foreigners',
            'history': 'Built so royal women could observe street life.',
            'facts': [
                'Has 953 windows.',
                'Designed to allow cool breeze inside.'
            ]
        },

        'Humayun\'s Tomb': {
            'name': "Humayun's Tomb",
            'location': 'Delhi',
            'description': 'Mausoleum of Mughal Emperor Humayun.',
            'built_year': '1570',
            'architect': 'Mirak Mirza Ghiyas',
            'significance': 'First garden-tomb in India.',
            'visiting_hours': '6:00 AM - 6:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹35 for Indians, ₹550 for foreigners',
            'history': 'Inspiration for Taj Mahal.',
            'facts': [
                'UNESCO Heritage Site.',
                'Built with red sandstone.'
            ]
        },

        'Iron Pillar': {
            'name': 'Iron Pillar',
            'location': 'Delhi',
            'description': '7-meter tall ancient iron pillar resistant to rust.',
            'built_year': '375-415 CE',
            'architect': 'Gupta Empire',
            'significance': 'Shows ancient metallurgical excellence.',
            'visiting_hours': '7:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹30 for Indians, ₹500 for foreigners',
            'history': 'Originally erected by King Chandragupta II.',
            'facts': [
                'Does not rust even after 1600 years.',
                'Located in Qutub complex.'
            ]
        },

        'Jamali Kamali Tomb': {
            'name': 'Jamali Kamali Tomb',
            'location': 'Mehrauli, Delhi',
            'description': 'Tomb of Sufi saints Jamali and Kamali.',
            'built_year': '1528-1536',
            'architect': 'Lodi-era builders',
            'significance': 'Important Sufi heritage site.',
            'visiting_hours': '9:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': 'Free',
            'history': 'Located in Mehrauli Archaeological Park.',
            'facts': [
                'Known for Persian inscriptions.',
                'Believed by locals to be haunted.'
            ]
        },

        'Khajuraho': {
            'name': 'Khajuraho Temples',
            'location': 'Chhatarpur, Madhya Pradesh',
            'description': 'Group of Hindu and Jain temples known for erotic sculptures.',
            'built_year': '950-1050 CE',
            'architect': 'Chandela dynasty',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '6:00 AM - 6:00 PM',
            'best_time': 'October to March',
            'entry_fee': '₹40 for Indians, ₹600 for foreigners',
            'history': 'Once had 85 temples, now only 25 survive.',
            'facts': [
                'Famous worldwide for stone carvings.',
                'Shows medieval Indian art at its peak.'
            ]
        },

        'Lotus Temple': {
            'name': 'Lotus Temple',
            'location': 'Delhi',
            'description': 'Bahá’í House of Worship shaped like a lotus flower.',
            'built_year': '1986',
            'architect': 'Fariborz Sahba',
            'significance': 'Symbol of peace and unity.',
            'visiting_hours': '9:00 AM - 5:00 PM',
            'best_time': 'October to March',
            'entry_fee': 'Free',
            'history': 'Visited by millions every year.',
            'facts': [
                'Made of 27 marble petals.',
                'No idols or rituals inside.'
            ]
        },

        'Mysore Palace': {
            'name': 'Mysore Palace',
            'location': 'Mysuru, Karnataka',
            'description': 'Royal residence of the Wadiyar dynasty.',
            'built_year': '1897-1912',
            'architect': 'Henry Irwin',
            'significance': 'Famous for Dussehra festival illumination.',
            'visiting_hours': '10:00 AM - 5:30 PM',
            'best_time': 'October to February',
            'entry_fee': '₹100 for Indians, ₹300 for foreigners',
            'history': 'Known for Indo-Saracenic architecture.',
            'facts': [
                'Lit with 97,000 bulbs during Dussehra.',
                'One of India’s most visited palaces.'
            ]
        },

        'Sun Temple Konark': {
            'name': 'Sun Temple Konark',
            'location': 'Konark, Odisha',
            'description': 'Temple shaped like a chariot dedicated to the Sun God.',
            'built_year': '1250 CE',
            'architect': 'King Narasimhadeva I',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '6:00 AM - 8:00 PM',
            'best_time': 'November to February',
            'entry_fee': '₹40 for Indians, ₹600 for foreigners',
            'history': 'Known for its stunning stone wheels.',
            'facts': [
                'Designed as a giant stone chariot.',
                'Iconic Kalinga architecture.'
            ]
        },

        'Thanjavur Temple': {
            'name': 'Thanjavur Temple',
            'location': 'Thanjavur, Tamil Nadu',
            'description': 'Brihadeeswarar Temple dedicated to Lord Shiva.',
            'built_year': '1010 CE',
            'architect': 'Rajaraja Chola I',
            'significance': 'UNESCO World Heritage Site.',
            'visiting_hours': '6:00 AM - 8:30 PM',
            'best_time': 'November to February',
            'entry_fee': 'Free',
            'history': 'Masterpiece of Chola architecture.',
            'facts': [
                'Temple tower is 66m tall.',
                'Made of granite blocks transported without wheels.'
            ]
        },

        'Victoria Memorial': {
            'name': 'Victoria Memorial',
            'location': 'Kolkata, West Bengal',
            'description': 'Huge marble museum dedicated to Queen Victoria.',
            'built_year': '1906-1921',
            'architect': 'William Emerson',
            'significance': 'Iconic heritage site of Kolkata.',
            'visiting_hours': '10:00 AM - 5:00 PM',
            'best_time': 'November to February',
            'entry_fee': '₹50 for Indians, ₹500 for foreigners',
            'history': 'Houses rare manuscripts, paintings, and artifacts.',
            'facts': [
                'Surrounded by beautiful gardens.',
                'Made entirely of white Makrana marble.'
            ]
        },
    }

    # 1. Direct key match (exact)
    if site_name in sites_info:
        return sites_info[site_name]

    # 2. Normalized match: remove spaces/underscores and lowercase
    def normalize(name):
        return name.replace(' ', '').replace('_', '').lower()

    normalized_target = normalize(site_name)
    for key, value in sites_info.items():
        if normalize(key) == normalized_target:
            return value

    # 3. Case-insensitive exact string match
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
    DATA_DIR = "dataset"
    
    # List of available heritage sites with detailed information
    all_sites = [
        {
            'name': 'Taj Mahal',
            'location': 'Agra, Uttar Pradesh',
            'built_year': '1632–1653',
            'image': 'Indian-monuments/images/train/tajmahal/2.jpg',
            'key': 'tajmahal',
        },
        {
            'name': 'Qutub Minar',
            'location': 'Delhi',
            'built_year': '1193',
            'image': 'Indian-monuments/images/train/qutub_minar/img346.jpg',
            'key': 'qutub_minar',
        },
        {
            'name': 'Gateway of India',
            'location': 'Mumbai, Maharashtra',
            'built_year': '1911–1924',
            'image': 'Indian-monuments/images/train/Gateway of India/1.jpg',
            'key': 'Gateway of India',
        },
        {
            "name": "Ajanta Caves",
            "location": "Aurangabad district, Maharashtra",
            "built_year": "2nd century BCE – 480 CE",
            "image": 'Indian-monuments/images/train/Ajanta Caves/(2).jpg',
            "key": "Ajanta Caves",
        },
        {
            "name": "Alai Darwaza",
            "location": "Delhi",
            "built_year": "1311",
            "image": 'Indian-monuments/images/train/alai_darwaza/img3.jpg',
            "key": "Alai Darwaza",
        },
        {
            "name": "Alai Minar",
            "location": "Delhi",
            "built_year": "14th century",
            "image": 'Indian-monuments/images/train/alai_minar/img3.jpg',
            "key": "Alai Minar",
        },
        {
            "name": "Basilica of Bom Jesus",
            "location": "Old Goa, Goa",
            "built_year": "1594–1605",
            "image": 'Indian-monuments/images/train/basilica_of_bom_jesus/1.jpg',
            "key": "Basilica of Bom Jesus",
        },
        {
            "name": "Charar-E-Sharif",
            "location": "Charar-e-Sharif, Jammu & Kashmir",
            "built_year": "15th century",
            "image": 'Indian-monuments/images/train/Charar-E- Sharif/51.jpg',
            "key": "Charar-E-Sharif",
        },
        {
            "name": "Charminar",
            "location": "Hyderabad, Telangana",
            "built_year": "1591",
            "image": 'Indian-monuments/images/train/charminar/(37).jpg',
            "key": "Charminar",
        },
        {
            "name": "Chhota Imambara",
            "location": "Lucknow, Uttar Pradesh",
            "built_year": "1838",
            "image": 'Indian-monuments/images/train/Chhota_Imambara/img1.jpg',
            "key": "Chhota Imambara",
        },
        {
            "name": "Ellora Caves",
            "location": "Aurangabad (Ellora), Maharashtra",
            "built_year": "600–1000 CE",
            "image": 'Indian-monuments/images/train/Ellora Caves/(2).jpg',
            "key": "Ellora Caves",
        },
        {
            "name": "Fatehpur Sikri",
            "location": "Near Agra, Uttar Pradesh",
            "built_year": "1571",
            "image": 'Indian-monuments/images/train/Fatehpur Sikri/1.jpg',
            "key": "Fatehpur Sikri",
        },
        {
            "name": "Golden Temple",
            "location": "Amritsar, Punjab",
            "built_year": "16th century",
            "image": 'Indian-monuments/images/train/golden temple/1.jpg',
            "key": "Golden Temple",
        },
        {
            "name": "Hawa Mahal",
            "location": "Jaipur, Rajasthan",
            "built_year": "1799",
            "image": 'Indian-monuments/images/train/hawa mahal pics/images-2.jpeg',
            "key": "Hawa Mahal",
        },
        {
            "name": "Humayun's Tomb",
            "location": "Delhi",
            "built_year": "1570",
            "image": 'Indian-monuments/images/train/Humayun_s Tomb/1.jpg',
            "key": "Humayun's Tomb",
        },
        {
            "name": "Iron Pillar",
            "location": "Delhi",
            "built_year": "4th century CE",
            "image": 'Indian-monuments/images/train/iron_pillar/img122.jpg',
            "key": "Iron Pillar",
        },
        {
            "name": "Jamali Kamali Tomb",
            "location": "Delhi",
            "built_year": "16th century",
            "image": 'Indian-monuments/images/train/jamali_kamali_tomb/img2.jpg',
            "key": "Jamali Kamali Tomb",
        },
        {
            "name": "Khajuraho",
            "location": "Khajuraho, Madhya Pradesh",
            "built_year": "950–1050 CE",
            "image": 'Indian-monuments/images/train/Khajuraho/img1.jpg',
            "key": "Khajuraho",
        },
        {
            "name": "Lotus Temple",
            "location": "Delhi",
            "built_year": "1986",
            "image": 'Indian-monuments/images/train/lotus_temple/18.jpg',
            "key": "Lotus Temple",
        },
        {
            "name": "Mysore Palace",
            "location": "Mysuru, Karnataka",
            "built_year": "1897–1912",
            "image": 'Indian-monuments/images/train/mysore_palace/1.jpg',
            "key": "Mysore Palace",
        },
        {
            "name": "Sun Temple Konark",
            "location": "Konark, Odisha",
            "built_year": "1250 CE",
            "image": 'Indian-monuments/images/train/Sun Temple Konark/(2).jpg',
            "key": "Sun Temple Konark",
        },
        {
            "name": "Thanjavur Temple",
            "location": "Thanjavur, Tamil Nadu",
            "built_year": "1010 CE",
            "image": 'Indian-monuments/images/train/tanjavur temple/1.jpg',
            "key": "Thanjavur Temple",
        },
        {
            "name": "Victoria Memorial",
            "location": "Kolkata, West Bengal",
            "built_year": "1906–1921",
            "image": 'Indian-monuments/images/train/victoria memorial/1.jpg',
            "key": "Victoria Memorial",
        },
    ]
    
    query = request.GET.get('q', '')

    if query:
        query_lower = query.lower()
        heritage_sites = [site for site in all_sites if query_lower in site['name'].lower() or query_lower in site['location'].lower()]
    else:
        heritage_sites = all_sites
    
    return render(request, 'search.html', {'heritage_sites': heritage_sites, 'query': query})


def site_detail(request, site_key):
    """
    Detailed information page for a single heritage site.
    Uses the existing get_site_info helper.
    """
    # Try to fetch detailed info by the exact key first
    site_info = get_site_info(site_key)

    # If not found, try a case-insensitive match on keys
    if not site_info:
        for key in [
            'tajmahal',
            'qutub_minar',
            'India Gate',
            'Gateway of India',
            'Ajanta Caves',
            'Alai Darwaza',
            'Alai Minar',
            'Basilica of Bom Jesus',
            'Charar-E-Sharif',
            'Charminar',
            'Chhota Imambara',
            'Ellora Caves',
            'Fatehpur Sikri',
            'Golden Temple',
            'Hawa Mahal',
            "Humayun's Tomb",
            'Iron Pillar',
            'Jamali Kamali Tomb',
            'Khajuraho',
            'Lotus Temple',
            'Mysore Palace',
            'Sun Temple Konark',
            'Thanjavur Temple',
            'Victoria Memorial',
        ]:
            if key.lower() == site_key.lower():
                site_info = get_site_info(key)
                break

    # Build a small gallery of images for this site from the dataset folder
    site_images = []
    if site_info:
        # Map normalized site names to folder names inside dataset/Indian-monuments/images/train
        def normalize(name):
            return name.replace(' ', '').replace('_', '').lower()

        folder_map = {
            'tajmahal': 'tajmahal',
            'qutubminar': 'qutub_minar',
            'indiagate': 'India gate pics',
            'gatewayofindia': 'Gateway of India',
            'ajantacaves': 'Ajanta Caves',
            'alaidarwaza': 'alai_darwaza',
            'alaiminar': 'alai_minar',
            'basilicaofbomjesus': 'basilica_of_bom_jesus',
            'charar-e-sharif': 'Charar-E- Sharif',
            'charminar': 'charminar',
            'chhotaimambara': 'Chhota_Imambara',
            'elloracaves': 'Ellora Caves',
            'fatehpursikri': 'Fatehpur Sikri',
            'goldentemple': 'golden temple',
            'hawamahal': 'hawa mahal pics',
            'humayunstomb': 'Humayun_s Tomb',
            'ironpillar': 'iron_pillar',
            'jamalikamaltomb': 'jamali_kamali_tomb',
            'khajuraho': 'Khajuraho',
            'lotustemple': 'lotus_temple',
            'mysorepalace': 'mysore_palace',
            'suntemplekonark': 'Sun Temple Konark',
            'thanjavurtemple': 'tanjavur temple',
            'victoriamemorial': 'victoria memorial',
        }

        normalized_key = normalize(site_key)
        folder = folder_map.get(normalized_key)

        if folder:
            # Prefer train images, fall back to test if needed
            base_dirs = [
                os.path.join(settings.BASE_DIR, 'dataset', 'Indian-monuments', 'images', 'train', folder),
                os.path.join(settings.BASE_DIR, 'dataset', 'Indian-monuments', 'images', 'test', folder),
            ]
            for base_dir in base_dirs:
                if os.path.isdir(base_dir):
                    files = sorted(os.listdir(base_dir))
                    # Keep only image-like files
                    files = [
                        f for f in files
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.webp'))
                    ]
                    for name in files[:3]:
                        rel_path = f'Indian-monuments/images/{os.path.basename(os.path.dirname(base_dir))}/{folder}/{name}'
                        site_images.append(rel_path)
                    if site_images:
                        break

    context = {
        'site_key': site_key,
        'site_info': site_info,
        'site_images': site_images,
    }
    return render(request, 'site_detail.html', context)

def guide(request):
    """Guide booking page"""
    from .models import GuideBooking
    from .forms import GuideBookingForm
    
    heritage_sites = [
    {'name': 'Taj Mahal', 'location': 'Agra, Uttar Pradesh'},
    {'name': 'Qutub Minar', 'location': 'Delhi'},
    {'name': 'Gateway of India', 'location': 'Mumbai, Maharashtra'},
    {'name': 'Ajanta Caves', 'location': 'Aurangabad, Maharashtra'},
    {'name': 'Alai Darwaza', 'location': 'Delhi'},
    {'name': 'Alai Minar', 'location': 'Delhi'},
    {'name': 'Basilica of Bom Jesus', 'location': 'Old Goa, Goa'},
    {'name': 'Charar-E-Sharif', 'location': 'Budgam, Jammu & Kashmir'},
    {'name': 'Charminar', 'location': 'Hyderabad, Telangana'},
    {'name': 'Chhota Imambara', 'location': 'Lucknow, Uttar Pradesh'},
    {'name': 'Ellora Caves', 'location': 'Aurangabad, Maharashtra'},
    {'name': 'Fatehpur Sikri', 'location': 'Agra, Uttar Pradesh'},
    {'name': 'Golden Temple', 'location': 'Amritsar, Punjab'},
    {'name': 'Hawa Mahal', 'location': 'Jaipur, Rajasthan'},
    {'name': 'Humayun\'s Tomb', 'location': 'Delhi'},
    {'name': 'Iron Pillar', 'location': 'Delhi'},
    {'name': 'Jamali Kamali Tomb', 'location': 'Mehrauli, Delhi'},
    {'name': 'Khajuraho', 'location': 'Chhatarpur, Madhya Pradesh'},
    {'name': 'Lotus Temple', 'location': 'Delhi'},
    {'name': 'Mysore Palace', 'location': 'Mysuru, Karnataka'},
    {'name': 'Sun Temple Konark', 'location': 'Konark, Odisha'},
    {'name': 'Thanjavur Temple', 'location': 'Thanjavur, Tamil Nadu'},
    {'name': 'Victoria Memorial', 'location': 'Kolkata, West Bengal'},
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
