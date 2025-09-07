# 🛒 Smart Grocery Reorder Assistant

An AI-powered grocery management system that solves real-world challenges faced by Indian consumers using services like Zepto, Blinkit, and other quick commerce platforms. This solution addresses delivery fee optimization, office ordering challenges, and smart inventory management.

## 🎯 Problem Solved

### Challenges Addressed:
- **Office Ordering Challenge**: When you're at office and don't remember what to order or quantities needed
- **Delivery Fee Optimization**: Avoiding multiple delivery charges by smart ordering and grace period management
- **Forgetting Items**: Missing items after placing an order, leading to additional delivery charges
- **Consumption Prediction**: Not knowing how much to order for your household
- **Seasonal Shopping**: Missing seasonal items and trending products

## ✨ Key Features

### 🤖 AI-Powered Intelligence
- **Machine Learning Predictions**: Random Forest Regressor for consumption forecasting
- **Smart Consumption Engine**: Indian household data-based consumption rates
- **Adaptive Learning**: AI learns from your order history and corrections
- **Custom Persona Support**: AI adapts to your unique household patterns

### 🛒 Smart Ordering System
- **Grace Period Management**: 30% of delivery time to add more items without extra charges
- **Real-time Order Tracking**: Live timers for grace period and delivery
- **Cart-based Ordering**: Multiple items in single order
- **Order History Management**: Complete order lifecycle tracking

### 📱 Advanced UI/UX
- **Responsive Design**: Works on all devices
- **Real-time Updates**: Live dashboard with current status
- **Smart Notifications**: Urgency-based alerts
- **Visual Order Status**: Clear indicators for active orders

### 🏠 Household Management
- **Multiple Persona Types**: Single, Couple, Family, Large Family, Hostel, Custom
- **Dynamic Household Detection**: AI learns actual household size from consumption
- **Guest Factor Adjustments**: Weekend (+30%), Holiday (+50%), Party (+100%)
- **Seasonal Recommendations**: Monsoon, Summer, Winter, Spring items

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd SmartGroceryAssistant
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

4. **Open your browser and go to:**
   ```
   http://localhost:8000
   ```

## 🏗️ Project Structure

```
SmartGroceryAssistant/
├── main.py                 # FastAPI backend with AI logic
├── requirements.txt        # Python dependencies
├── frontend/
│   └── index.html         # Modern React-like frontend
├── grocery_assistant.db   # SQLite database (created automatically)
└── README.md              # This file
```

## 🧠 How It Works

### AI Prediction Engine
The system uses a **Random Forest Regressor** with these features:
- Days since last purchase
- Daily consumption rate per person
- Current quantity available
- Household size and type
- Historical ordering patterns
- Guest factor adjustments

### Smart Consumption Engine
- **Base consumption rates** for Indian households
- **Category-specific patterns** (Dairy, Produce, Pantry, etc.)
- **Seasonal adjustments** for different times of year
- **Learning from corrections** when you delete orders

### Grace Period System
- **Dynamic calculation**: 30% of estimated delivery time
- **Real-time countdown**: Shows remaining grace time
- **Add items without extra charges** during grace period
- **Smart cancellation**: Restore suggestions if cancelled

## 📱 Using the Application

### 1. Dashboard
- **Total Items**: Current inventory count
- **Urgent Reorders**: Items needing immediate attention
- **AI Training Status**: Shows if AI is ready for predictions
- **Quick Actions**: Add items, view predictions, manage inventory

### 2. Persona Management
- **Predefined Personas**: Single, Couple, Family, Large Family, Hostel
- **Custom Personas**: Create personalized household profiles
- **AI Adaptation**: System learns your actual consumption patterns
- **Soft Delete/Reactivate**: Manage personas without losing data

### 3. Smart Ordering
- **Cart-based Orders**: Add multiple items and order together
- **Grace Period**: Add more items without extra delivery charges
- **Real-time Tracking**: Live timers for order status
- **Order History**: Complete order lifecycle management

### 4. AI Predictions
- **Urgency Levels**: High (< 3 days), Medium (3-7 days), Low (> 7 days)
- **Suggested Quantities**: AI-calculated optimal amounts
- **Seasonal Items**: Recommendations based on current season
- **Trending Products**: Popular items in your area

### 5. Order Management
- **Active Order Tracking**: Real-time status with timers
- **Order History**: Complete order records
- **Smart Cancellation**: Cancel from timer or order history
- **Suggestion Restoration**: Items reappear when orders cancelled

## 🔧 API Endpoints

### User Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/users/` | GET | List all active users |
| `/users/custom-persona/` | POST | Create custom persona |
| `/users/{id}/reactivate` | POST | Reactivate inactive user |
| `/users/{id}/permanent` | DELETE | Permanently delete user |

### Item Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/items/{user_id}` | GET | Get user's items |
| `/items/` | POST | Add new item |
| `/items/{id}` | PUT | Update item |
| `/items/{id}` | DELETE | Delete item |

### AI Predictions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predictions/{user_id}` | GET | Get AI predictions |
| `/order/{item_id}` | POST | Place individual order |
| `/users/{user_id}/learn-from-orders` | POST | Trigger AI learning |

### Order Management
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/orders/{user_id}/cart` | POST | Create cart order |
| `/orders/{order_id}/grace-period` | GET | Get grace period status |
| `/orders/{order_id}/end-grace-period` | POST | End grace period |
| `/order-history/{user_id}` | GET | Get order history |

### Smart Features
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze-cart/{user_id}` | POST | AI cart analysis |
| `/calculate-delivery-time` | POST | Delivery time calculation |
| `/delivery-tiers` | GET | Get delivery options |
| `/recommendations/{user_id}/seasonal` | GET | Seasonal recommendations |

## 🎯 Example Usage

### Creating Your First Order

1. **Add Items to Cart**:
   - Milk: 2 liters
   - Bread: 2 loaves
   - Eggs: 12 pieces

2. **Place Order**:
   - Grace period starts (e.g., 5 minutes)
   - Add more items if needed
   - Order confirmed after grace period

3. **Track Order**:
   - Real-time countdown timers
   - Delivery status updates
   - Cancel if needed

### AI Learning Example

1. **Order History**: System tracks your ordering patterns
2. **Consumption Learning**: AI calculates your actual usage
3. **Prediction Improvement**: Better suggestions over time
4. **Correction Learning**: Learns from deleted orders

## 🚨 Troubleshooting

### Common Issues

1. **Port 8000 already in use:**
   ```bash
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   ```

2. **Dependencies not found:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Database errors:**
   - Delete `grocery_assistant.db` file
   - Restart the application

4. **Timer not clearing:**
   - Check browser console for errors
   - Refresh the page
   - Clear browser cache

## 🔮 Future Enhancements

- **Multi-store Integration**: Compare prices across platforms
- **Meal Planning**: Recipe-based shopping suggestions
- **Health Tracking**: Dietary restriction filters
- **Social Features**: Family member notifications
- **Mobile App**: Native iOS/Android applications
- **Voice Commands**: Hands-free ordering
- **Price Tracking**: Historical price analysis

## 📊 Performance Metrics

- **AI Training**: ~2-3 seconds for 100+ items
- **Prediction Generation**: < 100ms per item
- **Database Queries**: Optimized with SQLAlchemy
- **Frontend Response**: < 200ms for most operations
- **Grace Period Accuracy**: ±1 minute precision

## 🏆 Unique Value Propositions

### For Indian Market
- **Localized Consumption Rates**: Based on Indian household data
- **Seasonal Awareness**: Monsoon, summer, winter recommendations
- **Delivery Fee Optimization**: Addresses quick commerce challenges
- **Office Ordering**: Solves workplace grocery management

### Technical Innovation
- **Grace Period System**: Unique to this solution
- **Real-time Order Tracking**: Live status updates
- **Smart Cancellation**: Flexible order management
- **AI Learning**: Continuous improvement from user behavior

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional ML algorithms
- Enhanced UI/UX features
- More delivery platform integrations
- Advanced analytics and reporting
- Mobile app development

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for Indian consumers who want smarter grocery management! 🛒✨**

*Solving real problems in the Indian quick commerce market with AI-powered intelligence.*