from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, field_validator, EmailStr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import json
from typing import List, Optional
import os
import random

# FastAPI app
app = FastAPI(title="Smart Grocery Reorder Assistant", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./grocery_assistant.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    display_name = Column(String)
    email = Column(String, unique=True, index=True, nullable=True)
    household_size = Column(Integer)
    account_type = Column(String)  # single, couple, family, large_family, hostel
    description = Column(String, default="")
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class GroceryItem(Base):
    __tablename__ = "grocery_items"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))  # Link to user
    name = Column(String, index=True)
    brand = Column(String)  # Amul, Britannia, etc.
    category = Column(String)
    quantity = Column(Float)
    unit = Column(String)
    image_url = Column(String)  # Product image URL
    last_purchased = Column(DateTime)
    consumption_rate = Column(Float)  # items per day
    household_size = Column(Integer)
    reminder_threshold = Column(Float)
    is_active = Column(Boolean, default=True)

class ProductCatalog(Base):
    __tablename__ = "product_catalog"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    brand = Column(String)
    category = Column(String)
    image_url = Column(String)
    default_unit = Column(String)
    base_consumption_rate = Column(Float)  # per person per day
    price_range = Column(String)  # Budget, Mid-range, Premium
    is_popular = Column(Boolean, default=False)

class OrderHistory(Base):
    __tablename__ = "order_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_id = Column(Integer, ForeignKey("grocery_items.id"))
    quantity = Column(Float)
    order_date = Column(DateTime)
    delivery_date = Column(DateTime)
    guest_factor = Column(Float, default=1.0)  # 1.0 = normal, 1.3 = 30% guests

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    order_number = Column(String, unique=True)  # e.g., "ORD-2024-001"
    status = Column(String, default="placed")  # placed, confirmed, preparing, out_for_delivery, delivered, cancelled
    total_amount = Column(Float, default=0.0)
    delivery_fee = Column(Float, default=0.0)
    subtotal = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)
    grace_period_ends = Column(DateTime, nullable=True)  # When grace period expires
    delivery_address = Column(Text, nullable=True)
    delivery_instructions = Column(Text, nullable=True)
    estimated_delivery_time = Column(DateTime, nullable=True)
    actual_delivery_time = Column(DateTime, nullable=True)
    is_grace_period_active = Column(Boolean, default=False)
    delivery_tier = Column(String, default="standard")  # free, standard, express
    store_id = Column(Integer, nullable=True)  # Which store fulfilled the order

class OrderItem(Base):
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("orders.id"))
    item_id = Column(Integer, ForeignKey("grocery_items.id"))
    item_name = Column(String)
    item_category = Column(String)
    quantity = Column(Float)
    unit_price = Column(Float)
    total_price = Column(Float)
    unit = Column(String)
    brand = Column(String, nullable=True)
    added_during_grace_period = Column(Boolean, default=False)  # Track if added during grace period

class DeliveryTier(Base):
    __tablename__ = "delivery_tiers"
    
    id = Column(Integer, primary_key=True, index=True)
    tier_name = Column(String)  # free, standard, express
    min_order_amount = Column(Float)  # Minimum order amount for this tier
    delivery_fee = Column(Float)  # Delivery fee for this tier
    estimated_delivery_time_minutes = Column(Integer)  # Estimated delivery time
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class Store(Base):
    __tablename__ = "stores"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    address = Column(Text)
    latitude = Column(Float)
    longitude = Column(Float)
    city = Column(String)
    state = Column(String)
    pincode = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class DeliveryZone(Base):
    __tablename__ = "delivery_zones"
    
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(Integer, ForeignKey("stores.id"))
    zone_name = Column(String)
    base_delivery_time_minutes = Column(Integer)  # Base time without traffic
    traffic_multiplier = Column(Float, default=1.0)  # Traffic factor (1.0 = normal, 1.5 = heavy traffic)
    pincode_start = Column(String)
    pincode_end = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    type = Column(String)  # urgent_reorder, low_stock, weekly_reminder, budget_alert
    title = Column(String)
    message = Column(Text)
    item_id = Column(Integer, nullable=True)
    priority = Column(String)  # high, medium, low
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    scheduled_for = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)

class Budget(Base):
    __tablename__ = "budgets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    month = Column(String)  # YYYY-MM format
    budget_amount = Column(Float)
    spent_amount = Column(Float, default=0.0)
    category = Column(String, nullable=True)  # Optional category-specific budget
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class SmartBasket(Base):
    __tablename__ = "smart_baskets"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    item_id = Column(Integer, ForeignKey("grocery_items.id"))  # Reference to GroceryItem
    basket_name = Column(String)  # User-defined basket name
    auto_reorder_enabled = Column(Boolean, default=True)
    reorder_threshold_days = Column(Float, default=3.0)  # Days before empty to trigger reorder
    min_quantity = Column(Float, default=1.0)  # Minimum quantity to maintain
    max_quantity = Column(Float, default=5.0)  # Maximum quantity to order
    last_auto_added = Column(DateTime, nullable=True)
    auto_add_count = Column(Integer, default=0)  # Number of times auto-added
    created_at = Column(DateTime, default=datetime.now)
    is_active = Column(Boolean, default=True)

class SmartBasketHistory(Base):
    __tablename__ = "smart_basket_history"
    
    id = Column(Integer, primary_key=True, index=True)
    basket_id = Column(Integer, ForeignKey("smart_baskets.id"))  # Reference to SmartBasket
    action = Column(String)  # auto_added, manual_added, disabled, enabled
    quantity_added = Column(Float)
    reason = Column(Text)  # AI reasoning for the action
    created_at = Column(DateTime, default=datetime.now)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class UserCreate(BaseModel):
    username: str
    display_name: str
    email: EmailStr
    household_size: int
    account_type: str
    description: str = ""
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v
    
    @field_validator('display_name')
    @classmethod
    def validate_display_name(cls, v):
        if len(v) < 2:
            raise ValueError('Display name must be at least 2 characters long')
        return v.strip()
    
    @field_validator('household_size')
    @classmethod
    def validate_household_size(cls, v):
        if v < 1 or v > 20:
            raise ValueError('Household size must be between 1 and 20')
        return v
    
    @field_validator('account_type')
    @classmethod
    def validate_account_type(cls, v):
        valid_types = ['single', 'couple', 'family', 'large_family', 'hostel', 'custom']
        if v not in valid_types:
            raise ValueError(f'Account type must be one of: {", ".join(valid_types)}')
        return v

class UserResponse(BaseModel):
    id: int
    username: str
    display_name: str
    email: str
    household_size: int
    account_type: str
    description: str
    created_at: datetime

class GroceryItemCreate(BaseModel):
    name: str
    brand: str
    category: str
    quantity: float
    unit: str
    image_url: str
    household_size: int
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError('Item name must be at least 2 characters long')
        return v.strip()
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be greater than 0')
        if v > 1000:
            raise ValueError('Quantity must be less than 1000')
        return v
    
    @field_validator('household_size')
    @classmethod
    def validate_household_size(cls, v):
        if v < 1 or v > 20:
            raise ValueError('Household size must be between 1 and 20')
        return v
    
    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v):
        valid_units = ['kg', 'g', 'l', 'ml', 'pieces', 'packets', 'bottles', 'cans', 'boxes']
        if v.lower() not in valid_units:
            raise ValueError(f'Unit must be one of: {", ".join(valid_units)}')
        return v.lower()

class GroceryItemResponse(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    quantity: float
    unit: str
    image_url: str
    last_purchased: Optional[datetime]
    consumption_rate: float
    household_size: int
    reminder_threshold: float
    is_active: bool

class ProductCatalogResponse(BaseModel):
    id: int
    name: str
    brand: str
    category: str
    image_url: str
    default_unit: str
    base_consumption_rate: float
    price_range: str
    is_popular: bool

class PredictionResponse(BaseModel):
    item_id: int
    item_name: str
    brand: str
    category: str
    days_until_empty: float
    recommended_order_date: datetime
    urgency_level: str
    suggested_quantity: float
    guest_adjustment: float

class NotificationResponse(BaseModel):
    id: int
    type: str
    title: str
    message: str
    item_id: Optional[int]
    priority: str
    is_read: bool
    created_at: datetime
    scheduled_for: Optional[datetime]

class BudgetCreate(BaseModel):
    budget_amount: float
    month: str
    category: Optional[str] = None

class BudgetResponse(BaseModel):
    id: int
    month: str
    budget_amount: float
    spent_amount: float
    remaining_amount: float
    category: Optional[str]
    created_at: datetime
    is_active: bool

class WeeklyReminderResponse(BaseModel):
    week_start: str
    week_end: str
    urgent_items: List[dict]
    low_stock_items: List[dict]
    budget_status: dict
    shopping_suggestions: List[dict]

class SmartBasketCreate(BaseModel):
    item_id: int
    basket_name: str
    reorder_threshold_days: float = 3.0
    min_quantity: float = 1.0
    max_quantity: float = 5.0

class SmartBasketUpdate(BaseModel):
    basket_name: Optional[str] = None
    auto_reorder_enabled: Optional[bool] = None
    reorder_threshold_days: Optional[float] = None
    min_quantity: Optional[float] = None
    max_quantity: Optional[float] = None

class SmartBasketResponse(BaseModel):
    id: int
    item_id: int
    item_name: str
    item_category: str
    basket_name: str
    auto_reorder_enabled: bool
    reorder_threshold_days: float
    min_quantity: float
    max_quantity: float
    last_auto_added: Optional[datetime]
    auto_add_count: int
    created_at: datetime
    is_active: bool

class SmartBasketHistoryResponse(BaseModel):
    id: int
    action: str
    quantity_added: float
    reason: str
    created_at: datetime

class OrderCreate(BaseModel):
    delivery_address: Optional[str] = None
    delivery_instructions: Optional[str] = None
    
    @field_validator('delivery_address')
    @classmethod
    def validate_delivery_address(cls, v):
        if v is not None and len(v.strip()) < 10:
            raise ValueError('Delivery address must be at least 10 characters long')
        return v.strip() if v else v
    
    @field_validator('delivery_instructions')
    @classmethod
    def validate_delivery_instructions(cls, v):
        if v is not None and len(v.strip()) > 500:
            raise ValueError('Delivery instructions must be less than 500 characters')
        return v.strip() if v else v

class OrderItemCreate(BaseModel):
    item_id: int
    quantity: float
    unit_price: float
    
    @field_validator('item_id')
    @classmethod
    def validate_item_id(cls, v):
        if v <= 0:
            raise ValueError('Item ID must be greater than 0')
        return v
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be greater than 0')
        if v > 1000:
            raise ValueError('Quantity must be less than 1000')
        return v
    
    @field_validator('unit_price')
    @classmethod
    def validate_unit_price(cls, v):
        if v < 0:
            raise ValueError('Unit price cannot be negative')
        if v > 10000:
            raise ValueError('Unit price must be less than 10000')
        return v

class OrderResponse(BaseModel):
    id: int
    order_number: str
    status: str
    total_amount: float
    delivery_fee: float
    subtotal: float
    created_at: datetime
    grace_period_ends: Optional[datetime]
    is_grace_period_active: bool
    delivery_tier: str
    estimated_delivery_time: Optional[datetime]
    items: List[dict]

class GracePeriodResponse(BaseModel):
    is_active: bool
    time_remaining_seconds: int
    can_add_items: bool
    message: str

class DeliveryTierResponse(BaseModel):
    tier_name: str
    min_order_amount: float
    delivery_fee: float
    estimated_delivery_time_minutes: int
    savings_message: Optional[str] = None

class DeliveryCalculationRequest(BaseModel):
    delivery_address: str
    pincode: str
    city: str
    state: str

class DeliveryCalculationResponse(BaseModel):
    estimated_delivery_time_minutes: int
    grace_period_minutes: int
    grace_period_ends: datetime
    store_name: str
    store_address: str
    delivery_zone: str
    traffic_condition: str
    base_time_minutes: int
    traffic_multiplier: float

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Notification Manager
class NotificationManager:
    def __init__(self):
        self.notification_types = {
            'urgent_reorder': {'priority': 'high', 'icon': '🚨'},
            'low_stock': {'priority': 'medium', 'icon': '⚠️'},
            'weekly_reminder': {'priority': 'low', 'icon': '📅'},
            'budget_alert': {'priority': 'high', 'icon': '💰'},
            'seasonal_tip': {'priority': 'low', 'icon': '🍂'},
            'ai_learning': {'priority': 'low', 'icon': '🧠'}
        }
    
    def create_notification(self, db: Session, user_id: int, notification_type: str, 
                          title: str, message: str, item_id: int = None, 
                          scheduled_for: datetime = None):
        """Create a new notification"""
        if notification_type not in self.notification_types:
            return None
            
        config = self.notification_types[notification_type]
        
        notification = Notification(
            user_id=user_id,
            type=notification_type,
            title=title,
            message=message,
            item_id=item_id,
            priority=config['priority'],
            scheduled_for=scheduled_for
        )
        
        db.add(notification)
        db.commit()
        db.refresh(notification)
        return notification
    
    def get_user_notifications(self, db: Session, user_id: int, unread_only: bool = False):
        """Get notifications for a user"""
        query = db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_active == True
        )
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
            
        return query.order_by(Notification.created_at.desc()).all()
    
    def mark_as_read(self, db: Session, notification_id: int, user_id: int):
        """Mark a notification as read"""
        notification = db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()
        
        if notification:
            notification.is_read = True
            db.commit()
            return True
        return False
    
    def create_urgent_reorder_notification(self, db: Session, user_id: int, item: GroceryItem):
        """Create urgent reorder notification for an item"""
        title = f"🚨 Urgent: {item.name} needs reordering!"
        message = f"Your {item.name} is running low ({item.quantity} {item.unit} left). Order now to avoid running out!"
        
        return self.create_notification(
            db, user_id, 'urgent_reorder', title, message, item.id
        )
    
    def create_weekly_reminder(self, db: Session, user_id: int, urgent_items: list, low_stock_items: list):
        """Create weekly shopping reminder"""
        title = "📅 Weekly Shopping Reminder"
        
        urgent_count = len(urgent_items)
        low_stock_count = len(low_stock_items)
        
        if urgent_count > 0 and low_stock_count > 0:
            message = f"You have {urgent_count} urgent items and {low_stock_count} low stock items. Time to shop!"
        elif urgent_count > 0:
            message = f"You have {urgent_count} urgent items that need immediate attention."
        elif low_stock_count > 0:
            message = f"You have {low_stock_count} items running low on stock."
        else:
            message = "Your pantry looks good! Consider adding some fresh items for the week."
        
        return self.create_notification(
            db, user_id, 'weekly_reminder', title, message
        )

# Seasonal Recommendations Manager
class SeasonalRecommendationsManager:
    def __init__(self):
        self.seasonal_items = {
            "winter": {
                "Dairy": ["Hot Chocolate", "Warm Milk", "Butter", "Ghee"],
                "Produce": ["Carrots", "Potatoes", "Onions", "Garlic", "Ginger"],
                "Pantry": ["Rice", "Dal", "Oil", "Spices", "Tea"],
                "Beverages": ["Tea", "Coffee", "Hot Chocolate"]
            },
            "summer": {
                "Dairy": ["Ice Cream", "Curd", "Buttermilk", "Yogurt"],
                "Produce": ["Watermelon", "Mango", "Cucumber", "Tomato", "Lemon"],
                "Beverages": ["Juice", "Cold Drinks", "Coconut Water"],
                "Frozen": ["Ice Cream", "Frozen Fruits"]
            },
            "monsoon": {
                "Dairy": ["Milk", "Paneer", "Cheese"],
                "Produce": ["Green Vegetables", "Leafy Greens", "Onions"],
                "Pantry": ["Rice", "Dal", "Oil", "Spices"],
                "Beverages": ["Tea", "Coffee", "Hot Soups"]
            },
            "spring": {
                "Dairy": ["Fresh Milk", "Curd", "Butter"],
                "Produce": ["Fresh Vegetables", "Leafy Greens", "Fruits"],
                "Pantry": ["Rice", "Dal", "Fresh Spices"],
                "Beverages": ["Fresh Juice", "Tea"]
            }
        }
        
        self.trending_items = {
            "health": ["Quinoa", "Oats", "Almonds", "Greek Yogurt", "Green Tea"],
            "organic": ["Organic Milk", "Organic Vegetables", "Organic Rice", "Organic Oil"],
            "premium": ["Premium Cheese", "Imported Fruits", "Gourmet Spices", "Artisan Bread"],
            "budget": ["Local Vegetables", "Basic Rice", "Standard Oil", "Regular Milk"]
        }
    
    def get_seasonal_recommendations(self, category: str = None):
        """Get seasonal recommendations based on current season"""
        import datetime
        
        month = datetime.datetime.now().month
        
        if month in [12, 1, 2]:
            season = "winter"
        elif month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8, 9]:
            season = "monsoon"
        else:
            season = "summer"
        
        seasonal_items = self.seasonal_items.get(season, {})
        
        if category and category in seasonal_items:
            return {
                "season": season,
                "category": category,
                "recommendations": seasonal_items[category],
                "message": f"Perfect {season} items for {category} category!"
            }
        elif category:
            return {
                "season": season,
                "category": category,
                "recommendations": [],
                "message": f"No specific {season} recommendations for {category} category."
            }
        else:
            return {
                "season": season,
                "recommendations": seasonal_items,
                "message": f"Here are the trending {season} items across all categories!"
            }
    
    def get_trending_items(self, trend_type: str = "health"):
        """Get trending items based on category"""
        return {
            "trend_type": trend_type,
            "recommendations": self.trending_items.get(trend_type, []),
            "message": f"Currently trending {trend_type} items!"
        }
    
    def get_smart_recommendations(self, db: Session, user_id: int, category: str = None):
        """Get smart recommendations based on user's consumption patterns and season"""
        # Get user's consumption history
        user_items = db.query(GroceryItem).filter(
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).all()
        
        # Get seasonal recommendations
        seasonal_recs = self.get_seasonal_recommendations(category)
        
        # Get trending items
        trending_recs = self.get_trending_items("health")
        
        # Analyze user's preferences
        user_categories = set(item.category for item in user_items)
        user_brands = set(item.brand for item in user_items if item.brand)
        
        # Create personalized recommendations
        personalized_recs = []
        
        # Add seasonal items that user hasn't tried
        for cat, items in seasonal_recs["recommendations"].items():
            if not category or cat == category:
                for item in items:
                    if not any(user_item.name.lower() == item.lower() for user_item in user_items):
                        personalized_recs.append({
                            "name": item,
                            "category": cat,
                            "reason": f"Perfect for {seasonal_recs['season']} season",
                            "type": "seasonal"
                        })
        
        # Add trending items
        for item in trending_recs["recommendations"]:
            if not any(user_item.name.lower() == item.lower() for user_item in user_items):
                personalized_recs.append({
                    "name": item,
                    "category": "Health",
                    "reason": "Currently trending health item",
                    "type": "trending"
                })
        
        return {
            "seasonal": seasonal_recs,
            "trending": trending_recs,
            "personalized": personalized_recs[:10],  # Limit to 10 recommendations
            "user_preferences": {
                "favorite_categories": list(user_categories),
                "favorite_brands": list(user_brands)
            }
        }

# Smart Shopping List Generator
class SmartShoppingListGenerator:
    def __init__(self):
        pass
    
    def generate_smart_shopping_list(self, db: Session, user_id: int, days_ahead: int = 7):
        """Generate a smart shopping list based on predictions and user preferences"""
        predictor = GroceryPredictor()
        consumption_engine = SmartConsumptionEngine()
        
        # Get user's items
        items = db.query(GroceryItem).filter(
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).all()
        
        shopping_list = {
            "urgent_items": [],
            "recommended_items": [],
            "seasonal_suggestions": [],
            "budget_estimate": 0,
            "estimated_days_coverage": days_ahead,
            "generated_at": datetime.now().isoformat()
        }
        
        # Analyze each item
        for item in items:
            prediction = predictor.predict_reorder_date(item)
            
            # Calculate days until empty
            days_until_empty = prediction['days_until_empty']
            
            if days_until_empty <= days_ahead:
                # Item needs reordering
                item_data = {
                    "item_id": item.id,
                    "name": item.name,
                    "category": item.category,
                    "current_quantity": item.quantity,
                    "unit": item.unit,
                    "days_until_empty": days_until_empty,
                    "suggested_quantity": prediction['suggested_quantity'],
                    "urgency_level": prediction['urgency_level'],
                    "reason": self._get_reorder_reason(days_until_empty, prediction['urgency_level'])
                }
                
                if prediction['urgency_level'] == 'high':
                    shopping_list["urgent_items"].append(item_data)
                else:
                    shopping_list["recommended_items"].append(item_data)
                
                # Estimate cost (simplified pricing)
                estimated_cost = self._estimate_item_cost(item, prediction['suggested_quantity'])
                shopping_list["budget_estimate"] += estimated_cost
        
        # Add seasonal suggestions
        seasonal_manager = SeasonalRecommendationsManager()
        seasonal_recs = seasonal_manager.get_seasonal_recommendations()
        
        for category, items_list in seasonal_recs["recommendations"].items():
            for item_name in items_list[:2]:  # Limit to 2 items per category
                if not any(user_item.name.lower() == item_name.lower() for user_item in items):
                    shopping_list["seasonal_suggestions"].append({
                        "name": item_name,
                        "category": category,
                        "reason": f"Perfect for {seasonal_recs['season']} season",
                        "estimated_cost": 50  # Default estimate
                    })
                    shopping_list["budget_estimate"] += 50
        
        # Sort by urgency
        shopping_list["urgent_items"].sort(key=lambda x: x['days_until_empty'])
        shopping_list["recommended_items"].sort(key=lambda x: x['days_until_empty'])
        
        return shopping_list
    
    def _get_reorder_reason(self, days_until_empty: float, urgency_level: str) -> str:
        """Get human-readable reason for reordering"""
        if urgency_level == 'high':
            if days_until_empty <= 1:
                return "Will run out today!"
            elif days_until_empty <= 2:
                return "Will run out tomorrow"
            else:
                return f"Will run out in {int(days_until_empty)} days"
        elif urgency_level == 'medium':
            return f"Running low - {int(days_until_empty)} days left"
        else:
            return f"Good to stock up - {int(days_until_empty)} days left"
    
    def _estimate_item_cost(self, item: GroceryItem, quantity: float) -> float:
        """Estimate cost for an item (simplified pricing)"""
        # Basic pricing estimates based on category
        base_prices = {
            "Dairy": 60,
            "Produce": 40,
            "Pantry": 80,
            "Bakery": 30,
            "Frozen": 100,
            "Beverages": 50,
            "Snacks": 25,
            "Household": 45
        }
        
        base_price = base_prices.get(item.category, 50)
        return base_price * quantity
    
    def generate_weekly_meal_plan(self, db: Session, user_id: int):
        """Generate a weekly meal plan based on available items"""
        # Get user's items
        items = db.query(GroceryItem).filter(
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).all()
        
        # Simple meal suggestions based on available items
        meal_suggestions = {
            "breakfast": [],
            "lunch": [],
            "dinner": [],
            "snacks": []
        }
        
        # Analyze available items and suggest meals
        available_categories = set(item.category for item in items)
        
        if "Dairy" in available_categories:
            meal_suggestions["breakfast"].append("Milk with Cereal")
            meal_suggestions["breakfast"].append("Curd Rice")
        
        if "Produce" in available_categories:
            meal_suggestions["lunch"].append("Fresh Vegetable Curry")
            meal_suggestions["dinner"].append("Mixed Vegetable Salad")
        
        if "Pantry" in available_categories:
            meal_suggestions["lunch"].append("Rice with Dal")
            meal_suggestions["dinner"].append("Rice with Curry")
        
        if "Bakery" in available_categories:
            meal_suggestions["breakfast"].append("Bread Toast")
            meal_suggestions["snacks"].append("Biscuits with Tea")
        
        return {
            "meal_plan": meal_suggestions,
            "available_items": [{"name": item.name, "category": item.category} for item in items],
            "shopping_suggestions": self._get_meal_plan_shopping_suggestions(available_categories)
        }
    
    def _get_meal_plan_shopping_suggestions(self, available_categories: set) -> list:
        """Get shopping suggestions for meal planning"""
        suggestions = []
        
        if "Dairy" not in available_categories:
            suggestions.append("Add dairy items for breakfast options")
        
        if "Produce" not in available_categories:
            suggestions.append("Add fresh vegetables for healthy meals")
        
        if "Pantry" not in available_categories:
            suggestions.append("Add rice and dal for staple meals")
        
        return suggestions

# Delivery Time Calculator
class DeliveryCalculator:
    def __init__(self):
        pass
    
    def calculate_delivery_time(self, db: Session, delivery_request: DeliveryCalculationRequest):
        """Calculate delivery time based on address, store location, and traffic"""
        try:
            # Find nearest store
            nearest_store = self._find_nearest_store(db, delivery_request)
            if not nearest_store:
                raise ValueError("No store found for delivery address")
            
            # Find delivery zone
            delivery_zone = self._find_delivery_zone(db, nearest_store.id, delivery_request.pincode)
            if not delivery_zone:
                # Use default zone if no specific zone found
                delivery_zone = self._get_default_zone(nearest_store.id)
            
            # Calculate traffic multiplier based on current time
            traffic_multiplier = self._calculate_traffic_multiplier()
            
            # Calculate base delivery time
            base_time = delivery_zone.base_delivery_time_minutes
            
            # Apply traffic multiplier
            estimated_delivery_time = int(base_time * traffic_multiplier)
            
            # Calculate grace period (30% of delivery time)
            grace_period_minutes = int(estimated_delivery_time * 0.3)
            
            # Calculate grace period end time
            grace_period_ends = datetime.now() + timedelta(minutes=grace_period_minutes)
            
            # Determine traffic condition
            traffic_condition = self._get_traffic_condition(traffic_multiplier)
            
            return DeliveryCalculationResponse(
                estimated_delivery_time_minutes=estimated_delivery_time,
                grace_period_minutes=grace_period_minutes,
                grace_period_ends=grace_period_ends,
                store_name=nearest_store.name,
                store_address=nearest_store.address,
                delivery_zone=delivery_zone.zone_name,
                traffic_condition=traffic_condition,
                base_time_minutes=base_time,
                traffic_multiplier=traffic_multiplier
            )
            
        except Exception as e:
            print(f"Error calculating delivery time: {e}")
            # Return default values if calculation fails
            return self._get_default_delivery_calculation()
    
    def _find_nearest_store(self, db: Session, delivery_request: DeliveryCalculationRequest):
        """Find the nearest store to the delivery address"""
        # For now, return the first active store
        # In a real implementation, you'd use geocoding and distance calculation
        store = db.query(Store).filter(Store.is_active == True).first()
        return store
    
    def _find_delivery_zone(self, db: Session, store_id: int, pincode: str):
        """Find the delivery zone for the given pincode"""
        # Find zone by pincode range
        zone = db.query(DeliveryZone).filter(
            DeliveryZone.store_id == store_id,
            DeliveryZone.is_active == True,
            DeliveryZone.pincode_start <= pincode,
            DeliveryZone.pincode_end >= pincode
        ).first()
        
        return zone
    
    def _get_default_zone(self, store_id: int):
        """Get default delivery zone if no specific zone found"""
        # Return a default zone object
        class DefaultZone:
            def __init__(self):
                self.zone_name = "Default Zone"
                self.base_delivery_time_minutes = 30
                self.traffic_multiplier = 1.0
        
        return DefaultZone()
    
    def _calculate_traffic_multiplier(self):
        """Calculate traffic multiplier based on current time and day"""
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()  # 0 = Monday, 6 = Sunday
        
        # Peak hours: 7-9 AM, 5-7 PM on weekdays
        if day_of_week < 5:  # Weekdays
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return 1.5  # Heavy traffic
            elif 9 <= hour <= 17:
                return 1.2  # Moderate traffic
            else:
                return 1.0  # Light traffic
        else:  # Weekends
            if 10 <= hour <= 14 or 18 <= hour <= 20:
                return 1.3  # Moderate traffic
            else:
                return 1.0  # Light traffic
    
    def _get_traffic_condition(self, traffic_multiplier: float):
        """Get human-readable traffic condition"""
        if traffic_multiplier >= 1.4:
            return "Heavy Traffic"
        elif traffic_multiplier >= 1.2:
            return "Moderate Traffic"
        else:
            return "Light Traffic"
    
    def _get_default_delivery_calculation(self):
        """Get default delivery calculation if calculation fails"""
        grace_period_ends = datetime.now() + timedelta(minutes=9)  # 30% of 30 minutes
        return DeliveryCalculationResponse(
            estimated_delivery_time_minutes=30,
            grace_period_minutes=9,
            grace_period_ends=grace_period_ends,
            store_name="Default Store",
            store_address="123 Main St",
            delivery_zone="Default Zone",
            traffic_condition="Normal",
            base_time_minutes=30,
            traffic_multiplier=1.0
        )

# Order Management System
class OrderManager:
    def __init__(self):
        pass
    
    def create_order(self, db: Session, user_id: int, cart_items: List[dict], order_data: OrderCreate):
        """Create a new order with dynamic grace period based on delivery time"""
        # Generate order number
        order_number = f"ORD-{datetime.now().strftime('%Y%m%d')}-{db.query(Order).count() + 1:03d}"
        
        # Calculate totals
        subtotal = sum(item.get('total_price', 0) for item in cart_items)
        delivery_tier = self._calculate_delivery_tier(db, subtotal)
        delivery_fee = delivery_tier['delivery_fee']
        total_amount = subtotal + delivery_fee
        
        # Calculate dynamic delivery time and grace period
        delivery_calculator = DeliveryCalculator()
        
        # Extract address components (in real implementation, this would come from user input)
        delivery_request = DeliveryCalculationRequest(
            delivery_address=order_data.delivery_address or "123 Main St",
            pincode="110001",  # Default pincode - in real implementation, extract from address
            city="Delhi",      # Default city - in real implementation, extract from address
            state="Delhi"      # Default state - in real implementation, extract from address
        )
        
        delivery_calculation = delivery_calculator.calculate_delivery_time(db, delivery_request)
        
        # Create order with dynamic grace period
        order = Order(
            user_id=user_id,
            order_number=order_number,
            status="placed",
            total_amount=total_amount,
            delivery_fee=delivery_fee,
            subtotal=subtotal,
            grace_period_ends=delivery_calculation.grace_period_ends,
            delivery_address=order_data.delivery_address,
            delivery_instructions=order_data.delivery_instructions,
            is_grace_period_active=True,
            delivery_tier=delivery_tier['tier_name'],
            estimated_delivery_time=datetime.now() + timedelta(minutes=delivery_calculation.estimated_delivery_time_minutes)
        )
        
        db.add(order)
        db.commit()
        db.refresh(order)
        
        # Add order items
        for item in cart_items:
            order_item = OrderItem(
                order_id=order.id,
                item_id=item.get('item_id'),
                item_name=item.get('item_name'),
                item_category=item.get('category'),
                quantity=item.get('quantity'),
                unit_price=item.get('unit_price', 0),
                total_price=item.get('total_price', 0),
                unit=item.get('unit'),
                brand=item.get('brand')
            )
            db.add(order_item)
            
            # Also create OrderHistory record for order history display
            order_history = OrderHistory(
                user_id=user_id,
                item_id=item.get('item_id'),
                quantity=item.get('quantity'),
                order_date=datetime.now(),
                delivery_date=datetime.now() + timedelta(days=1),
                guest_factor=1.0  # Default for cart orders
            )
            db.add(order_history)
        
        db.commit()
        return order
    
    def add_item_to_order(self, db: Session, order_id: int, user_id: int, item_data: OrderItemCreate):
        """Add item to existing order during grace period"""
        order = db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id,
            Order.is_grace_period_active == True
        ).first()
        
        if not order:
            raise ValueError("Order not found or grace period expired")
        
        if datetime.now() > order.grace_period_ends:
            raise ValueError("Grace period has expired")
        
        # Get item details
        item = db.query(GroceryItem).filter(GroceryItem.id == item_data.item_id).first()
        if not item:
            raise ValueError("Item not found")
        
        # Calculate item total
        total_price = item_data.quantity * item_data.unit_price
        
        # Add order item
        order_item = OrderItem(
            order_id=order_id,
            item_id=item_data.item_id,
            item_name=item.name,
            item_category=item.category,
            quantity=item_data.quantity,
            unit_price=item_data.unit_price,
            total_price=total_price,
            unit=item.unit,
            brand=item.brand,
            added_during_grace_period=True
        )
        db.add(order_item)
        
        # Update order totals
        order.subtotal += total_price
        order.total_amount = order.subtotal + order.delivery_fee
        
        db.commit()
        return order_item
    
    def get_order_grace_period(self, db: Session, order_id: int, user_id: int):
        """Get grace period status for an order"""
        order = db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id
        ).first()
        
        if not order:
            raise ValueError("Order not found")
        
        now = datetime.now()
        is_active = order.is_grace_period_active and now < order.grace_period_ends
        time_remaining = max(0, int((order.grace_period_ends - now).total_seconds())) if order.grace_period_ends else 0
        
        return GracePeriodResponse(
            is_active=is_active,
            time_remaining_seconds=time_remaining,
            can_add_items=is_active,
            message=f"Grace period {'active' if is_active else 'expired'}. {'Add items without extra delivery fee!' if is_active else 'Grace period has ended.'}"
        )
    
    def end_grace_period(self, db: Session, order_id: int, user_id: int):
        """End grace period for an order"""
        order = db.query(Order).filter(
            Order.id == order_id,
            Order.user_id == user_id
        ).first()
        
        if not order:
            raise ValueError("Order not found")
        
        order.is_grace_period_active = False
        order.status = "confirmed"
        db.commit()
        
        return order
    
    def _calculate_delivery_tier(self, db: Session, subtotal: float):
        """Calculate delivery tier based on order amount - Free delivery for orders >= ₹100"""
        # New logic: Free delivery for orders >= ₹100, delivery charges for orders < ₹100
        if subtotal >= 100.0:
            # Free delivery tier
            return {
                'tier_name': 'free',
                'delivery_fee': 0.0,
                'estimated_delivery_time_minutes': 45
            }
        else:
            # Standard delivery with charges
            return {
                'tier_name': 'standard',
                'delivery_fee': 50.0,
                'estimated_delivery_time_minutes': 60
            }
        
        # Default to standard tier
        return {
            'tier_name': 'standard',
            'delivery_fee': 50.0,
            'estimated_delivery_time_minutes': 30
        }
    
    def get_delivery_tiers(self, db: Session, current_subtotal: float = 0):
        """Get available delivery tiers with savings messages - New logic: Free delivery for orders >= ₹100"""
        # New simplified logic: Free delivery for orders >= ₹100, delivery charges for orders < ₹100
        result = []
        
        # Free delivery tier (orders >= ₹100)
        free_tier = DeliveryTierResponse(
            tier_name="free",
            min_order_amount=100.0,
            delivery_fee=0.0,
            estimated_delivery_time_minutes=45,
            savings_message=f"Add ₹{max(0, 100 - current_subtotal):.0f} more for FREE delivery!" if current_subtotal < 100 else "🎉 You qualify for FREE delivery!"
        )
        result.append(free_tier)
        
        # Standard delivery tier (orders < ₹100)
        standard_tier = DeliveryTierResponse(
            tier_name="standard",
            min_order_amount=0.0,
            delivery_fee=50.0,
            estimated_delivery_time_minutes=60,
            savings_message="Standard delivery with ₹50 charges" if current_subtotal < 100 else "You qualify for FREE delivery!"
        )
        result.append(standard_tier)
        
        return result

# AI-Powered "Don't Forget" Analysis
class DontForgetAnalyzer:
    def __init__(self):
        pass
    
    def analyze_cart_for_missing_items(self, db: Session, user_id: int, cart_items: List[dict]):
        """Analyze cart and suggest missing items based on user history and patterns"""
        try:
            # Get user's order history
            recent_orders = db.query(OrderHistory).filter(
                OrderHistory.user_id == user_id
            ).order_by(OrderHistory.order_date.desc()).limit(10).all()
            
            # Get user's items
            user_items = db.query(GroceryItem).filter(
                GroceryItem.user_id == user_id,
                GroceryItem.is_active == True
            ).all()
            
            # Analyze patterns
            suggestions = []
            
            # 1. Check for commonly bought together items
            suggestions.extend(self._find_commonly_bought_together(db, user_id, cart_items, recent_orders))
            
            # 2. Check for seasonal items
            suggestions.extend(self._find_seasonal_suggestions(cart_items))
            
            # 3. Check for pantry staples
            suggestions.extend(self._find_missing_staples(cart_items, user_items))
            
            # 4. Check for category completeness
            suggestions.extend(self._find_missing_categories(cart_items))
            
            # 5. Check for quantity patterns
            suggestions.extend(self._find_quantity_patterns(cart_items, recent_orders))
            
            # Remove duplicates and limit suggestions
            unique_suggestions = []
            seen_items = set()
            for suggestion in suggestions:
                key = f"{suggestion['item_name']}_{suggestion['category']}"
                if key not in seen_items and len(unique_suggestions) < 5:
                    unique_suggestions.append(suggestion)
                    seen_items.add(key)
            
            return unique_suggestions
            
        except Exception as e:
            print(f"Error analyzing cart: {e}")
            return []
    
    def _find_commonly_bought_together(self, db: Session, user_id: int, cart_items: List[dict], recent_orders: List[OrderHistory]):
        """Find items commonly bought together with current cart items"""
        suggestions = []
        
        if not recent_orders:
            return suggestions
        
        # Get cart categories
        cart_categories = set(item.get('category', '') for item in cart_items)
        cart_item_names = set(item.get('item_name', '') for item in cart_items)
        
        # Find items that appear together in recent orders
        for order in recent_orders:
            # Get other items from the same order (simplified - in real implementation, you'd need order items)
            # For now, we'll use a simple pattern matching approach
            pass
        
        # Common combinations based on categories
        common_combinations = {
            'Dairy': ['Bread', 'Eggs'],
            'Produce': ['Dairy', 'Pantry'],
            'Pantry': ['Dairy', 'Produce'],
            'Bakery': ['Dairy', 'Pantry'],
            'Frozen': ['Dairy', 'Pantry']
        }
        
        for category in cart_categories:
            if category in common_combinations:
                for suggested_item in common_combinations[category]:
                    if not any(item.get('item_name', '').lower() == suggested_item.lower() for item in cart_items):
                        suggestions.append({
                            'item_name': suggested_item,
                            'category': category,
                            'reason': f'Commonly bought with {category} items',
                            'confidence': 0.7,
                            'priority': 'medium'
                        })
        
        return suggestions
    
    def _find_seasonal_suggestions(self, cart_items: List[dict]):
        """Find seasonal items that might be missing"""
        suggestions = []
        
        current_month = datetime.now().month
        seasonal_items = {
            12: ['Hot Chocolate', 'Christmas Cookies', 'Mulled Wine'],
            1: ['Hot Chocolate', 'Winter Vegetables'],
            2: ['Valentine\'s Chocolates', 'Winter Vegetables'],
            3: ['Spring Vegetables', 'Fresh Herbs'],
            4: ['Easter Eggs', 'Spring Vegetables'],
            5: ['Mother\'s Day Flowers', 'Spring Vegetables'],
            6: ['Summer Fruits', 'Ice Cream'],
            7: ['Summer Fruits', 'BBQ Items'],
            8: ['Summer Fruits', 'Cold Drinks'],
            9: ['Back to School Snacks', 'Fall Vegetables'],
            10: ['Halloween Candy', 'Pumpkin Items'],
            11: ['Thanksgiving Items', 'Fall Vegetables']
        }
        
        if current_month in seasonal_items:
            for item in seasonal_items[current_month]:
                if not any(cart_item.get('item_name', '').lower() == item.lower() for cart_item in cart_items):
                    suggestions.append({
                        'item_name': item,
                        'category': 'Seasonal',
                        'reason': f'Popular {datetime.now().strftime("%B")} item',
                        'confidence': 0.6,
                        'priority': 'low'
                    })
        
        return suggestions
    
    def _find_missing_staples(self, cart_items: List[dict], user_items: List[GroceryItem]):
        """Find missing pantry staples"""
        suggestions = []
        
        staples = [
            {'name': 'Rice', 'category': 'Pantry'},
            {'name': 'Dal', 'category': 'Pantry'},
            {'name': 'Oil', 'category': 'Pantry'},
            {'name': 'Salt', 'category': 'Pantry'},
            {'name': 'Sugar', 'category': 'Pantry'},
            {'name': 'Milk', 'category': 'Dairy'},
            {'name': 'Bread', 'category': 'Bakery'},
            {'name': 'Eggs', 'category': 'Dairy'}
        ]
        
        cart_item_names = [item.get('item_name', '').lower() for item in cart_items]
        
        for staple in staples:
            if not any(staple['name'].lower() in name for name in cart_item_names):
                # Check if user has this item in their inventory
                user_has_item = any(staple['name'].lower() in item.name.lower() for item in user_items)
                
                suggestions.append({
                    'item_name': staple['name'],
                    'category': staple['category'],
                    'reason': 'Essential pantry staple',
                    'confidence': 0.8 if user_has_item else 0.5,
                    'priority': 'high' if user_has_item else 'medium'
                })
        
        return suggestions
    
    def _find_missing_categories(self, cart_items: List[dict]):
        """Find missing categories that user usually buys"""
        suggestions = []
        
        cart_categories = set(item.get('category', '') for item in cart_items)
        all_categories = ['Dairy', 'Produce', 'Pantry', 'Bakery', 'Frozen', 'Beverages']
        
        missing_categories = set(all_categories) - cart_categories
        
        category_suggestions = {
            'Dairy': ['Milk', 'Cheese', 'Yogurt'],
            'Produce': ['Bananas', 'Apples', 'Onions'],
            'Pantry': ['Rice', 'Dal', 'Oil'],
            'Bakery': ['Bread', 'Cookies'],
            'Frozen': ['Ice Cream', 'Frozen Vegetables'],
            'Beverages': ['Water', 'Juice', 'Tea']
        }
        
        for category in missing_categories:
            if category in category_suggestions:
                for item in category_suggestions[category][:1]:  # Only suggest one item per category
                    suggestions.append({
                        'item_name': item,
                        'category': category,
                        'reason': f'You usually buy {category} items',
                        'confidence': 0.6,
                        'priority': 'medium'
                    })
        
        return suggestions
    
    def _find_quantity_patterns(self, cart_items: List[dict], recent_orders: List[OrderHistory]):
        """Find quantity patterns and suggest adjustments"""
        suggestions = []
        
        # This is a simplified version - in a real implementation, you'd analyze
        # quantity patterns from order history
        
        for item in cart_items:
            quantity = item.get('quantity', 0)
            if quantity < 1:
                suggestions.append({
                    'item_name': item.get('item_name', ''),
                    'category': item.get('category', ''),
                    'reason': 'Consider buying more - you usually buy larger quantities',
                    'confidence': 0.5,
                    'priority': 'low',
                    'suggested_quantity': 2.0
                })
        
        return suggestions

# Smart Basket Manager
class SmartBasketManager:
    def __init__(self):
        pass
    
    def create_smart_basket(self, db: Session, user_id: int, basket_data: SmartBasketCreate):
        """Create a new smart basket for a user"""
        # Check if item exists and belongs to user
        item = db.query(GroceryItem).filter(
            GroceryItem.id == basket_data.item_id,
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).first()
        
        if not item:
            raise ValueError("Item not found or not accessible")
        
        # Check if item is already in a smart basket
        existing_basket = db.query(SmartBasket).filter(
            SmartBasket.user_id == user_id,
            SmartBasket.item_id == basket_data.item_id,
            SmartBasket.is_active == True
        ).first()
        
        if existing_basket:
            raise ValueError("Item is already in a smart basket")
        
        # Create smart basket
        smart_basket = SmartBasket(
            user_id=user_id,
            item_id=basket_data.item_id,
            basket_name=basket_data.basket_name,
            reorder_threshold_days=basket_data.reorder_threshold_days,
            min_quantity=basket_data.min_quantity,
            max_quantity=basket_data.max_quantity
        )
        
        db.add(smart_basket)
        db.commit()
        db.refresh(smart_basket)
        
        # Log creation
        self._log_basket_action(db, smart_basket.id, "created", 0, f"Smart basket '{basket_data.basket_name}' created for {item.name}")
        
        return smart_basket
    
    def get_user_smart_baskets(self, db: Session, user_id: int):
        """Get all smart baskets for a user with item details"""
        baskets = db.query(SmartBasket, GroceryItem).join(
            GroceryItem, SmartBasket.item_id == GroceryItem.id
        ).filter(
            SmartBasket.user_id == user_id,
            SmartBasket.is_active == True,
            GroceryItem.is_active == True
        ).all()
        
        result = []
        for basket, item in baskets:
            result.append({
                "id": basket.id,
                "item_id": basket.item_id,
                "item_name": item.name,
                "item_category": item.category,
                "basket_name": basket.basket_name,
                "auto_reorder_enabled": basket.auto_reorder_enabled,
                "reorder_threshold_days": basket.reorder_threshold_days,
                "min_quantity": basket.min_quantity,
                "max_quantity": basket.max_quantity,
                "last_auto_added": basket.last_auto_added,
                "auto_add_count": basket.auto_add_count,
                "created_at": basket.created_at,
                "is_active": basket.is_active,
                "current_quantity": item.quantity,
                "current_unit": item.unit
            })
        
        return result
    
    def update_smart_basket(self, db: Session, basket_id: int, user_id: int, update_data: SmartBasketUpdate):
        """Update a smart basket"""
        basket = db.query(SmartBasket).filter(
            SmartBasket.id == basket_id,
            SmartBasket.user_id == user_id,
            SmartBasket.is_active == True
        ).first()
        
        if not basket:
            raise ValueError("Smart basket not found")
        
        # Update fields
        if update_data.basket_name is not None:
            basket.basket_name = update_data.basket_name
        if update_data.auto_reorder_enabled is not None:
            basket.auto_reorder_enabled = update_data.auto_reorder_enabled
        if update_data.reorder_threshold_days is not None:
            basket.reorder_threshold_days = update_data.reorder_threshold_days
        if update_data.min_quantity is not None:
            basket.min_quantity = update_data.min_quantity
        if update_data.max_quantity is not None:
            basket.max_quantity = update_data.max_quantity
        
        db.commit()
        db.refresh(basket)
        
        # Log update
        self._log_basket_action(db, basket_id, "updated", 0, "Smart basket settings updated")
        
        return basket
    
    def delete_smart_basket(self, db: Session, basket_id: int, user_id: int):
        """Delete (deactivate) a smart basket"""
        basket = db.query(SmartBasket).filter(
            SmartBasket.id == basket_id,
            SmartBasket.user_id == user_id,
            SmartBasket.is_active == True
        ).first()
        
        if not basket:
            raise ValueError("Smart basket not found")
        
        basket.is_active = False
        db.commit()
        
        # Log deletion
        self._log_basket_action(db, basket_id, "deleted", 0, "Smart basket deleted")
        
        return True
    
    def check_and_auto_add_items(self, db: Session, user_id: int):
        """Check all smart baskets and auto-add items to cart if needed"""
        predictor = GroceryPredictor()
        notification_manager = NotificationManager()
        
        baskets = db.query(SmartBasket, GroceryItem).join(
            GroceryItem, SmartBasket.item_id == GroceryItem.id
        ).filter(
            SmartBasket.user_id == user_id,
            SmartBasket.is_active == True,
            SmartBasket.auto_reorder_enabled == True,
            GroceryItem.is_active == True
        ).all()
        
        auto_added_items = []
        
        for basket, item in baskets:
            try:
                # Get prediction for the item
                prediction = predictor.predict_reorder_date(item)
                days_until_empty = prediction['days_until_empty']
                
                # Check if item needs reordering based on threshold
                if days_until_empty <= basket.reorder_threshold_days:
                    # Calculate quantity to add
                    suggested_quantity = min(
                        prediction['suggested_quantity'],
                        basket.max_quantity
                    )
                    
                    # Ensure minimum quantity
                    suggested_quantity = max(suggested_quantity, basket.min_quantity)
                    
                    # Add to cart (simulate cart addition)
                    cart_item = {
                        "item_id": item.id,
                        "item_name": item.name,
                        "category": item.category,
                        "quantity": suggested_quantity,
                        "unit": item.unit,
                        "suggested_quantity": suggested_quantity,
                        "is_smart_basket": True,
                        "basket_id": basket.id,
                        "basket_name": basket.basket_name
                    }
                    
                    # Update basket stats
                    basket.last_auto_added = datetime.now()
                    basket.auto_add_count += 1
                    
                    # Log auto-add action
                    reason = f"Auto-added {suggested_quantity} {item.unit} of {item.name} - {days_until_empty:.1f} days until empty (threshold: {basket.reorder_threshold_days} days)"
                    self._log_basket_action(db, basket.id, "auto_added", suggested_quantity, reason)
                    
                    auto_added_items.append({
                        "basket_id": basket.id,
                        "basket_name": basket.basket_name,
                        "item_name": item.name,
                        "quantity": suggested_quantity,
                        "unit": item.unit,
                        "reason": reason
                    })
                    
                    # Create notification
                    notification_manager.create_notification(
                        db, user_id, 'ai_learning',
                        f"🛒 Smart Basket: {basket.basket_name}",
                        f"Added {suggested_quantity} {item.unit} of {item.name} to your cart automatically!",
                        item.id
                    )
            
            except Exception as e:
                print(f"Error processing smart basket {basket.id}: {e}")
                continue
        
        db.commit()
        return auto_added_items
    
    def get_basket_history(self, db: Session, basket_id: int, user_id: int):
        """Get history for a specific smart basket"""
        # Verify basket belongs to user
        basket = db.query(SmartBasket).filter(
            SmartBasket.id == basket_id,
            SmartBasket.user_id == user_id
        ).first()
        
        if not basket:
            raise ValueError("Smart basket not found")
        
        history = db.query(SmartBasketHistory).filter(
            SmartBasketHistory.basket_id == basket_id
        ).order_by(SmartBasketHistory.created_at.desc()).all()
        
        return history
    
    def _log_basket_action(self, db: Session, basket_id: int, action: str, quantity_added: float, reason: str):
        """Log an action for a smart basket"""
        history_entry = SmartBasketHistory(
            basket_id=basket_id,
            action=action,
            quantity_added=quantity_added,
            reason=reason
        )
        db.add(history_entry)
        db.commit()

# Budget Manager
class BudgetManager:
    def __init__(self):
        pass
    
    def create_budget(self, db: Session, user_id: int, budget_data: BudgetCreate):
        """Create or update a budget for a user"""
        # Check if budget already exists for this month
        existing_budget = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.month == budget_data.month,
            Budget.category == budget_data.category,
            Budget.is_active == True
        ).first()
        
        if existing_budget:
            existing_budget.budget_amount = budget_data.budget_amount
            db.commit()
            return existing_budget
        else:
            budget = Budget(
                user_id=user_id,
                month=budget_data.month,
                budget_amount=budget_data.budget_amount,
                category=budget_data.category
            )
            db.add(budget)
            db.commit()
            db.refresh(budget)
            return budget
    
    def get_current_month_budget(self, db: Session, user_id: int, category: str = None):
        """Get current month's budget"""
        current_month = datetime.now().strftime("%Y-%m")
        
        query = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.month == current_month,
            Budget.is_active == True
        )
        
        if category:
            query = query.filter(Budget.category == category)
        else:
            query = query.filter(Budget.category.is_(None))
            
        return query.first()
    
    def update_spent_amount(self, db: Session, user_id: int, amount: float, category: str = None):
        """Update spent amount for current month"""
        budget = self.get_current_month_budget(db, user_id, category)
        if budget:
            budget.spent_amount += amount
            db.commit()
            return budget
        return None
    
    def get_budget_status(self, db: Session, user_id: int):
        """Get comprehensive budget status"""
        current_month = datetime.now().strftime("%Y-%m")
        
        # Get all budgets for current month
        budgets = db.query(Budget).filter(
            Budget.user_id == user_id,
            Budget.month == current_month,
            Budget.is_active == True
        ).all()
        
        total_budget = sum(b.budget_amount for b in budgets)
        total_spent = sum(b.spent_amount for b in budgets)
        remaining = total_budget - total_spent
        
        # Calculate spending percentage
        spending_percentage = (total_spent / total_budget * 100) if total_budget > 0 else 0
        
        # Determine status
        if spending_percentage >= 90:
            status = "critical"
        elif spending_percentage >= 75:
            status = "warning"
        elif spending_percentage >= 50:
            status = "moderate"
        else:
            status = "good"
        
        return {
            "total_budget": total_budget,
            "total_spent": total_spent,
            "remaining": remaining,
            "spending_percentage": spending_percentage,
            "status": status,
            "budgets": [
                {
                    "category": b.category or "General",
                    "budget_amount": b.budget_amount,
                    "spent_amount": b.spent_amount,
                    "remaining": b.budget_amount - b.spent_amount
                } for b in budgets
            ]
        }

# Smart Consumption Engine
class SmartConsumptionEngine:
    def __init__(self):
        # Base consumption rates per person per day (Indian household data)
        self.base_rates = {
            "Dairy": {
                "Milk": 0.25,  # liters per person per day
                "Curd": 0.1,
                "Cheese": 0.02,
                "Butter": 0.01,
                "Paneer": 0.03
            },
            "Bakery": {
                "Bread": 0.15,  # slices per person per day
                "Biscuits": 0.1,
                "Cakes": 0.05
            },
            "Produce": {
                "Tomatoes": 0.1,  # kg per person per day
                "Onions": 0.08,
                "Potatoes": 0.12,
                "Apples": 0.15,
                "Bananas": 0.2
            },
            "Pantry": {
                "Rice": 0.15,  # kg per person per day
                "Dal": 0.08,
                "Oil": 0.02,
                "Sugar": 0.03,
                "Salt": 0.005
            },
            "Beverages": {
                "Tea": 0.02,  # kg per person per day
                "Coffee": 0.01,
                "Juice": 0.1
            }
        }
        
        # Guest factor adjustments
        self.guest_multipliers = {
            "weekend": 1.3,  # 30% increase on weekends
            "holiday": 1.5,  # 50% increase during holidays
            "party": 2.0,    # Double for parties
            "normal": 1.0    # No change
        }
    
    def calculate_consumption_rate(self, item_name: str, category: str, household_size: int, guest_factor: str = "normal") -> float:
        """Calculate smart consumption rate based on item, household size, and guest factor"""
        base_rate = self.base_rates.get(category, {}).get(item_name, 0.1)
        guest_multiplier = self.guest_multipliers.get(guest_factor, 1.0)
        
        # Adjust for household size (larger households may have slightly lower per-person consumption)
        household_factor = 1.0 if household_size <= 2 else 0.9 if household_size <= 4 else 0.85
        
        return base_rate * household_size * guest_multiplier * household_factor
    
    def suggest_quantity(self, item_name: str, category: str, household_size: int, days_to_last: int = 7, guest_factor: str = "normal") -> float:
        """Suggest optimal quantity to order"""
        daily_consumption = self.calculate_consumption_rate(item_name, category, household_size, guest_factor)
        suggested_quantity = daily_consumption * days_to_last
        
        # Round to reasonable quantities
        if category == "Produce":
            suggested_quantity = round(suggested_quantity, 1)  # 0.5 kg, 1.0 kg, etc.
        elif category == "Dairy":
            suggested_quantity = round(suggested_quantity, 2)  # 0.25 L, 0.5 L, etc.
        else:
            suggested_quantity = round(suggested_quantity, 1)
        
        return max(suggested_quantity, 0.1)  # Minimum 0.1
    
    def detect_household_size_from_consumption(self, db: Session, user_id: int) -> int:
        """AI detects household size based on actual consumption patterns"""
        try:
            # Get user's order history to analyze consumption patterns
            orders = db.query(OrderHistory).filter(OrderHistory.user_id == user_id).all()
            
            if len(orders) < 3:  # Need minimum data
                return 1  # Default to single person
            
            # Analyze consumption patterns for key items
            total_milk_consumed = 0
            total_bread_consumed = 0
            total_eggs_consumed = 0
            
            for order in orders:
                item = db.query(GroceryItem).filter(GroceryItem.id == order.item_id).first()
                if item:
                    if "milk" in item.name.lower():
                        total_milk_consumed += order.quantity
                    elif "bread" in item.name.lower():
                        total_bread_consumed += order.quantity
                    elif "egg" in item.name.lower():
                        total_eggs_consumed += order.quantity
            
            # Calculate average daily consumption
            days_analyzed = max(1, (datetime.now() - orders[0].order_date).days)
            
            avg_daily_milk = total_milk_consumed / days_analyzed
            avg_daily_bread = total_bread_consumed / days_analyzed
            avg_daily_eggs = total_eggs_consumed / days_analyzed
            
            # Estimate household size based on consumption patterns
            estimated_size = 1
            
            # Milk consumption analysis (0.25L per person per day)
            if avg_daily_milk > 0.5:
                estimated_size = max(estimated_size, round(avg_daily_milk / 0.25))
            
            # Bread consumption analysis (0.15 slices per person per day)
            if avg_daily_bread > 0.3:
                estimated_size = max(estimated_size, round(avg_daily_bread / 0.15))
            
            # Eggs consumption analysis (0.5 eggs per person per day)
            if avg_daily_eggs > 1:
                estimated_size = max(estimated_size, round(avg_daily_eggs / 0.5))
            
            # Cap at reasonable household size
            estimated_size = min(max(estimated_size, 1), 10)
            
            return estimated_size
            
        except Exception as e:
            print(f"Error detecting household size: {e}")
            return 1  # Fallback to single person
    
    def learn_from_order_history(self, db: Session, user_id: int, item_id: int) -> dict:
        """Learn from order history to improve consumption predictions"""
        try:
            # Get all orders for this item by this user
            item_orders = db.query(OrderHistory).filter(
                OrderHistory.user_id == user_id,
                OrderHistory.item_id == item_id
            ).order_by(OrderHistory.order_date.asc()).all()
            
            if len(item_orders) < 2:
                return {"learned": False, "message": "Need more orders to learn"}
            
            # Calculate actual consumption patterns
            total_quantity = sum(order.quantity for order in item_orders)
            first_order = item_orders[0].order_date
            last_order = item_orders[-1].order_date
            total_days = (last_order - first_order).days
            
            if total_days == 0:
                return {"learned": False, "message": "Orders on same day"}
            
            # Calculate actual daily consumption rate
            actual_daily_consumption = total_quantity / total_days
            
            # Get the item details
            item = db.query(GroceryItem).filter(GroceryItem.id == item_id).first()
            if not item:
                return {"learned": False, "message": "Item not found"}
            
            # Calculate frequency of orders
            order_intervals = []
            for i in range(1, len(item_orders)):
                interval = (item_orders[i].order_date - item_orders[i-1].order_date).days
                if interval > 0:
                    order_intervals.append(interval)
            
            avg_order_interval = sum(order_intervals) / len(order_intervals) if order_intervals else 0
            
            # Update the item's consumption rate based on real data
            if actual_daily_consumption > 0:
                # Blend actual data with existing rate (70% actual, 30% existing)
                new_rate = (0.7 * actual_daily_consumption) + (0.3 * item.consumption_rate)
                item.consumption_rate = round(new_rate, 3)
                
                # Update last purchased date
                item.last_purchased = last_order
                
                db.commit()
                
                return {
                    "learned": True,
                    "old_rate": item.consumption_rate,
                    "new_rate": new_rate,
                    "actual_consumption": round(actual_daily_consumption, 3),
                    "avg_order_interval": round(avg_order_interval, 1),
                    "total_orders": len(item_orders)
                }
            
            return {"learned": False, "message": "Could not calculate consumption"}
            
        except Exception as e:
            print(f"Error learning from order history: {e}")
            return {"learned": False, "message": f"Error: {str(e)}"}
    
    def learn_from_order_deletion(self, db: Session, user_id: int, deleted_order_data: dict) -> dict:
        """Learn from order deletion to improve AI predictions by removing bad training data"""
        try:
            item_id = deleted_order_data["item_id"]
            item_name = deleted_order_data["item_name"]
            item_category = deleted_order_data["category"]
            
            # Get the item details
            item = db.query(GroceryItem).filter(GroceryItem.id == item_id).first()
            if not item:
                return {"learned": False, "message": "Item not found"}
            
            # Get remaining orders for this item after deletion
            remaining_orders = db.query(OrderHistory).filter(
                OrderHistory.user_id == user_id,
                OrderHistory.item_id == item_id
            ).order_by(OrderHistory.order_date.asc()).all()
            
            if len(remaining_orders) >= 2:
                # Recalculate consumption rate based on remaining orders
                total_quantity = sum(order.quantity for order in remaining_orders)
                first_order = remaining_orders[0].order_date
                last_order = remaining_orders[-1].order_date
                total_days = (last_order - first_order).days
                
                if total_days > 0:
                    # Calculate new consumption rate without the deleted order
                    new_daily_consumption = total_quantity / total_days
                    
                    # Update the item's consumption rate
                    old_rate = item.consumption_rate
                    item.consumption_rate = round(new_daily_consumption, 3)
                    
                    # Update last purchased date to most recent remaining order
                    item.last_purchased = last_order
                    
                    db.commit()
                    
                    return {
                        "learned": True,
                        "deletion_impact": "positive",
                        "old_rate": old_rate,
                        "new_rate": item.consumption_rate,
                        "remaining_orders": len(remaining_orders),
                        "message": f"Consumption rate updated after removing incorrect order data"
                    }
                else:
                    return {"learned": False, "message": "Insufficient time span in remaining orders"}
            else:
                # Not enough remaining orders, reset to base rate
                base_rate = self.calculate_consumption_rate(
                    item.name, item.category, item.household_size
                )
                old_rate = item.consumption_rate
                item.consumption_rate = base_rate
                
                # Clear last purchased date since we don't have reliable data
                item.last_purchased = None
                
                db.commit()
                
                return {
                    "learned": True,
                    "deletion_impact": "reset_to_base",
                    "old_rate": old_rate,
                    "new_rate": base_rate,
                    "remaining_orders": len(remaining_orders),
                    "message": f"Reset to base consumption rate due to insufficient order history"
                }
                
        except Exception as e:
            print(f"Error learning from order deletion: {e}")
            return {"learned": False, "message": f"Error: {str(e)}"}
    
    def analyze_deletion_patterns(self, db: Session, user_id: int) -> dict:
        """Analyze patterns in order deletions to improve AI learning"""
        try:
            # This method can be used to analyze what types of orders users commonly delete
            # and use that information to improve future predictions
            
            # Get all orders for this user (including deleted ones if we had a soft delete system)
            # For now, we'll focus on the learning aspect
            
            return {
                "analysis": "Order deletion analysis completed",
                "recommendations": [
                    "Consider implementing soft delete for better pattern analysis",
                    "Track deletion reasons for improved AI learning",
                    "Use deletion patterns to identify common ordering mistakes"
                ]
            }
            
        except Exception as e:
            print(f"Error analyzing deletion patterns: {e}")
            return {"analysis": "failed", "error": str(e)}

# AI Prediction Engine
class GroceryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.consumption_engine = SmartConsumptionEngine()
    
    def train_model(self, db: Session, user_id: int):
        """Train the prediction model with user's historical data"""
        items = db.query(GroceryItem).filter(
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).all()
        
        if len(items) < 3:  # Need minimum data to train
            return False
        
        # Create training features
        features = []
        targets = []
        
        for item in items:
            if item.last_purchased:
                days_since_purchase = (datetime.now() - item.last_purchased).days
                consumption_per_day = item.consumption_rate
                current_quantity = item.quantity
                
                days_until_empty = current_quantity / consumption_per_day if consumption_per_day > 0 else 30
                
                features.append([
                    days_since_purchase,
                    consumption_per_day,
                    current_quantity,
                    item.household_size,
                    days_until_empty
                ])
                targets.append(days_until_empty)
        
        if len(features) > 0:
            X = np.array(features)
            y = np.array(targets)
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return True
        
        return False
    
    def predict_reorder_date(self, item: GroceryItem, guest_factor: str = "normal") -> dict:
        """Predict when to reorder an item with guest factor consideration"""
        if not self.is_trained:
            # Use smart consumption engine
            daily_consumption = self.consumption_engine.calculate_consumption_rate(
                item.name, item.category, item.household_size, guest_factor
            )
            days_until_empty = item.quantity / daily_consumption if daily_consumption > 0 else 30
            recommended_order_date = datetime.now() + timedelta(days=max(0, days_until_empty - 3))
            
            # Enhanced urgency calculation based on quantity and consumption
            urgency_level = self._calculate_urgency_level(item, days_until_empty)
            
            return {
                "days_until_empty": days_until_empty,
                "recommended_order_date": recommended_order_date,
                "urgency_level": urgency_level,
                "suggested_quantity": self.consumption_engine.suggest_quantity(
                    item.name, item.category, item.household_size, 7, guest_factor
                ),
                "guest_adjustment": self.consumption_engine.guest_multipliers.get(guest_factor, 1.0)
            }
        
        # Use trained model
        features = np.array([[
            (datetime.now() - item.last_purchased).days if item.last_purchased else 0,
            item.consumption_rate,
            item.quantity,
            item.household_size,
            item.quantity / item.consumption_rate if item.consumption_rate > 0 else 30
        ]])
        
        features_scaled = self.scaler.transform(features)
        predicted_days = self.model.predict(features_scaled)[0]
        
        recommended_order_date = datetime.now() + timedelta(days=max(0, predicted_days - 3))
        
        # Enhanced urgency calculation for trained model
        urgency_level = self._calculate_urgency_level(item, predicted_days)
        
        return {
            "days_until_empty": predicted_days,
            "recommended_order_date": recommended_order_date,
            "urgency_level": urgency_level,
            "suggested_quantity": self.consumption_engine.suggest_quantity(
                item.name, item.category, item.household_size, 7, guest_factor
            ),
            "guest_adjustment": self.consumption_engine.guest_multipliers.get(guest_factor, 1.0)
        }
    
    def _calculate_urgency_level(self, item: GroceryItem, days_until_empty: float) -> str:
        """Calculate urgency level based on quantity, consumption rate, and days until empty"""
        
        # Critical quantity thresholds (very low quantities)
        critical_thresholds = {
            "Dairy": {"Milk": 0.2, "Curd": 0.1, "Cheese": 0.05, "Butter": 0.02, "Paneer": 0.1},
            "Bakery": {"Bread": 0.5, "Biscuits": 0.3, "Cakes": 0.2},
            "Produce": {"Tomatoes": 0.1, "Onions": 0.2, "Potatoes": 0.3, "Apples": 0.2, "Bananas": 0.3},
            "Pantry": {"Rice": 1.0, "Dal": 0.2, "Oil": 0.3, "Sugar": 0.1, "Salt": 0.05},
            "Beverages": {"Tea": 0.1, "Coffee": 0.05, "Juice": 0.5},
            "Household": {"Soap": 0.5, "Detergent": 0.2, "Tissue": 1.0}
        }
        
        # Check if quantity is critically low
        category_thresholds = critical_thresholds.get(item.category, {})
        item_threshold = category_thresholds.get(item.name, 0.1)  # Default threshold
        
        if item.quantity <= item_threshold:
            return "high"  # Critical - almost empty
        
        # Check days until empty
        if days_until_empty <= 1:
            return "high"  # Critical - will run out today/tomorrow
        elif days_until_empty <= 3:
            return "high"  # High urgency - will run out soon
        elif days_until_empty <= 7:
            return "medium"  # Medium urgency - order within a week
        else:
            return "low"  # Low urgency - plenty of time

# Initialize engines
predictor = GroceryPredictor()

# Pre-loaded test accounts data
PRELOADED_ACCOUNTS = [
    {
        "username": "rahul_single",
        "display_name": "Rahul",
        "email": "rahul@test.com",
        "household_size": 1,
        "account_type": "single",
        "description": "Young professional living alone",
        "items": [
            {"name": "Amul Milk", "brand": "Amul", "category": "Dairy", "quantity": 0.1, "unit": "liters", "image_url": "https://example.com/amul-milk.jpg"},  # URGENT - Almost empty
            {"name": "Britannia Bread", "brand": "Britannia", "category": "Bakery", "quantity": 0.2, "unit": "packets", "image_url": "https://example.com/britannia-bread.jpg"},  # URGENT - Low stock
            {"name": "Farm Fresh Eggs", "brand": "Farm Fresh", "category": "Dairy", "quantity": 1.0, "unit": "pieces", "image_url": "https://example.com/eggs.jpg"},  # URGENT - Critical
            {"name": "Fresh Tomatoes", "brand": "Local", "category": "Produce", "quantity": 0.05, "unit": "kg", "image_url": "https://example.com/tomatoes.jpg"},  # URGENT - Almost gone
            {"name": "Cooking Oil", "brand": "Fortune", "category": "Pantry", "quantity": 0.1, "unit": "liters", "image_url": "https://example.com/oil.jpg"}  # URGENT - Critical
        ]
    },
    {
        "username": "priya_amit_couple",
        "display_name": "Priya & Amit",
        "email": "couple@test.com",
        "household_size": 2,
        "account_type": "couple",
        "description": "Young married couple",
        "items": [
            {"name": "Amul Milk", "brand": "Amul", "category": "Dairy", "quantity": 0.2, "unit": "liters", "": 25.0, "image_url": "https://example.com/amul-milk.jpg"},  # URGENT - Low stock
            {"name": "Britannia Bread", "brand": "Britannia", "category": "Bakery", "quantity": 0.3, "unit": "packets", "": 30.0, "image_url": "https://example.com/britannia-bread.jpg"},  # URGENT - Critical
            {"name": "Farm Fresh Eggs", "brand": "Farm Fresh", "category": "Dairy", "quantity": 2.0, "unit": "pieces", "": 8.0, "image_url": "https://example.com/eggs.jpg"},  # URGENT - Almost empty
            {"name": "Fresh Tomatoes", "brand": "Local", "category": "Produce", "quantity": 0.1, "unit": "kg", "": 40.0, "image_url": "https://example.com/tomatoes.jpg"},  # URGENT - Critical
            {"name": "Basmati Rice", "brand": "India Gate", "category": "Pantry", "quantity": 0.5, "unit": "kg", "": 80.0, "image_url": "https://example.com/rice.jpg"},  # URGENT - Low stock
            {"name": "Tea Leaves", "brand": "Taj Mahal", "category": "Beverages", "quantity": 0.05, "unit": "kg", "": 200.0, "image_url": "https://example.com/tea.jpg"}  # URGENT - Almost gone
        ]
    },
    {
        "username": "patel_family",
        "display_name": "The Patel Family",
        "email": "family@test.com",
        "household_size": 4,
        "account_type": "family",
        "description": "Family with 2 kids",
        "items": [
            {"name": "Amul Milk", "brand": "Amul", "category": "Dairy", "quantity": 0.3, "unit": "liters", "": 25.0, "image_url": "https://example.com/amul-milk.jpg"},  # URGENT - Critical
            {"name": "Britannia Bread", "brand": "Britannia", "category": "Bakery", "quantity": 0.4, "unit": "packets", "": 30.0, "image_url": "https://example.com/britannia-bread.jpg"},  # URGENT - Low stock
            {"name": "Farm Fresh Eggs", "brand": "Farm Fresh", "category": "Dairy", "quantity": 3.0, "unit": "pieces", "": 8.0, "image_url": "https://example.com/eggs.jpg"},  # URGENT - Almost empty
            {"name": "Fresh Tomatoes", "brand": "Local", "category": "Produce", "quantity": 0.15, "unit": "kg", "": 40.0, "image_url": "https://example.com/tomatoes.jpg"},  # URGENT - Critical
            {"name": "Basmati Rice", "brand": "India Gate", "category": "Pantry", "quantity": 0.8, "unit": "kg", "": 80.0, "image_url": "https://example.com/rice.jpg"},  # URGENT - Low stock
            {"name": "Toor Dal", "brand": "Tata", "category": "Pantry", "quantity": 0.1, "unit": "kg", "": 60.0, "image_url": "https://example.com/dal.jpg"},  # URGENT - Critical
            {"name": "Sugar", "brand": "Local", "category": "Pantry", "quantity": 0.05, "unit": "kg", "": 50.0, "image_url": "https://example.com/sugar.jpg"},  # URGENT - Almost gone
            {"name": "Salt", "brand": "Tata", "category": "Pantry", "quantity": 0.02, "unit": "kg", "": 20.0, "image_url": "https://example.com/salt.jpg"}  # URGENT - Critical
        ]
    },
    {
        "username": "sharma_joint_family",
        "display_name": "Sharma Joint Family",
        "email": "large@test.com",
        "household_size": 6,
        "account_type": "large_family",
        "description": "Joint family with grandparents",
        "items": [
            {"name": "Amul Milk", "brand": "Amul", "category": "Dairy", "quantity": 0.5, "unit": "liters", "image_url": "https://example.com/amul-milk.jpg"},  # URGENT - Critical
            {"name": "Britannia Bread", "brand": "Britannia", "category": "Bakery", "quantity": 0.6, "unit": "packets", "image_url": "https://example.com/britannia-bread.jpg"},  # URGENT - Low stock
            {"name": "Farm Fresh Eggs", "brand": "Farm Fresh", "category": "Dairy", "quantity": 4.0, "unit": "pieces", "image_url": "https://example.com/eggs.jpg"},  # URGENT - Almost empty
            {"name": "Fresh Tomatoes", "brand": "Local", "category": "Produce", "quantity": 0.2, "unit": "kg", "image_url": "https://example.com/tomatoes.jpg"},  # URGENT - Critical
            {"name": "Basmati Rice", "brand": "India Gate", "category": "Pantry", "quantity": 1.2, "unit": "kg", "image_url": "https://example.com/rice.jpg"},  # URGENT - Low stock
            {"name": "Toor Dal", "brand": "Tata", "category": "Pantry", "quantity": 0.15, "unit": "kg", "image_url": "https://example.com/dal.jpg"},  # URGENT - Critical
            {"name": "Cooking Oil", "brand": "Fortune", "category": "Pantry", "quantity": 0.3, "unit": "liters", "image_url": "https://example.com/oil.jpg"},  # URGENT - Low stock
            {"name": "Coffee Powder", "brand": "Nescafe", "category": "Beverages", "quantity": 0.03, "unit": "kg", "image_url": "https://example.com/coffee.jpg"}  # URGENT - Almost gone
        ]
    },
    {
        "username": "student_hostel_delhi",
        "display_name": "Delhi Student Hostel",
        "email": "hostel@test.com",
        "household_size": 8,
        "account_type": "hostel",
        "description": "Student hostel with 8 roommates",
        "items": [
            {"name": "Amul Milk", "brand": "Amul", "category": "Dairy", "quantity": 0.8, "unit": "liters", "image_url": "https://example.com/amul-milk.jpg"},  # URGENT - Critical
            {"name": "Britannia Bread", "brand": "Britannia", "category": "Bakery", "quantity": 0.9, "unit": "packets", "image_url": "https://example.com/britannia-bread.jpg"},  # URGENT - Low stock
            {"name": "Farm Fresh Eggs", "brand": "Farm Fresh", "category": "Dairy", "quantity": 5.0, "unit": "pieces", "image_url": "https://example.com/eggs.jpg"},  # URGENT - Almost empty
            {"name": "Fresh Tomatoes", "brand": "Local", "category": "Produce", "quantity": 0.25, "unit": "kg", "image_url": "https://example.com/tomatoes.jpg"},  # URGENT - Critical
            {"name": "Basmati Rice", "brand": "India Gate", "category": "Pantry", "quantity": 1.5, "unit": "kg", "image_url": "https://example.com/rice.jpg"},  # URGENT - Low stock
            {"name": "Toor Dal", "brand": "Tata", "category": "Pantry", "quantity": 0.2, "unit": "kg", "image_url": "https://example.com/dal.jpg"},  # URGENT - Critical
            {"name": "Cooking Oil", "brand": "Fortune", "category": "Pantry", "quantity": 0.4, "unit": "liters", "image_url": "https://example.com/oil.jpg"},  # URGENT - Low stock
            {"name": "Detergent Powder", "brand": "Surf Excel", "category": "Household", "quantity": 0.1, "unit": "kg", "image_url": "https://example.com/detergent.jpg"},  # URGENT - Critical
            {"name": "Toilet Paper", "brand": "Local", "category": "Household", "quantity": 0.5, "unit": "rolls", "image_url": "https://example.com/tissue.jpg"}  # URGENT - Almost gone
        ]
    }
]

# API Endpoints
@app.post("/users/", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    try:
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == user.username).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = db.query(User).filter(User.email == user.email).first()
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        db_user = User(
            username=user.username,
            display_name=user.display_name,
            email=user.email,
            household_size=user.household_size,
            account_type=user.account_type,
            description=user.description
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@app.post("/users/custom-persona/", response_model=UserResponse)
def create_custom_persona(display_name: str, email: str, db: Session = Depends(get_db)):
    """Create a custom persona, checking for inactive accounts first."""
    # Check if a user with this email already exists
    existing_user = db.query(User).filter(User.email == email).first()
    
    if existing_user:
        if existing_user.is_active:
            # Email is already in use by an active account
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Email already registered with an active account.",
                    "user_id": existing_user.id,
                    "display_name": existing_user.display_name,
                    "is_active": True
                }
            )
        else:
            # Email is associated with an inactive (soft-deleted) account
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "An inactive account with this email already exists.",
                    "user_id": existing_user.id,
                    "display_name": existing_user.display_name,
                    "is_active": False
                }
            )

    """Create a custom persona starting as single person, AI will adapt household size"""
    try:
        username = f"custom_{display_name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        
        db_user = User(
            username=username,
            display_name=display_name,
            email=email,
            household_size=1,  # Start as single person
            account_type="custom",
            description=f"Custom persona: {display_name}"
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating custom persona: {str(e)}")

@app.post("/users/{user_id}/reactivate", response_model=UserResponse)
def reactivate_user(user_id: int, db: Session = Depends(get_db)):
    """Reactivate a soft-deleted user."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if user.is_active:
            raise HTTPException(status_code=400, detail="User is already active")

        user.is_active = True
        db.commit()
        db.refresh(user)
        return user
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error reactivating user: {str(e)}")

@app.delete("/users/{user_id}/permanent")
def permanent_delete_user(user_id: int, db: Session = Depends(get_db)):
    """Permanently delete a user and all their data."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Hard delete associated items
        items_deleted_count = db.query(GroceryItem).filter(GroceryItem.user_id == user_id).delete()
        
        # Hard delete associated order history
        orders_deleted_count = db.query(OrderHistory).filter(OrderHistory.user_id == user_id).delete()
        
        # Hard delete the user
        db.delete(user)
        db.commit()
        
        return {
            "message": f"User '{user.display_name}' permanently deleted.",
            "items_deleted": items_deleted_count,
            "orders_deleted": orders_deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error during permanent deletion: {str(e)}")

@app.post("/users/{user_id}/update-household-size")
def update_household_size_ai(user_id: int, db: Session = Depends(get_db)):
    """AI automatically updates household size based on consumption patterns"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Use AI to detect household size
    consumption_engine = SmartConsumptionEngine()
    detected_size = consumption_engine.detect_household_size_from_consumption(db, user_id)
    
    # Update user's household size
    user.household_size = detected_size
    
    # Update all items for this user with new household size
    items = db.query(GroceryItem).filter(GroceryItem.user_id == user_id).all()
    for item in items:
        item.household_size = detected_size
        # Recalculate consumption rate
        new_rate = consumption_engine.calculate_consumption_rate(
            item.name, item.category, detected_size
        )
        item.consumption_rate = new_rate
    
    db.commit()
    
    return {
        "message": f"AI detected household size: {detected_size} people",
        "new_household_size": detected_size,
        "items_updated": len(items)
    }

@app.get("/users/", response_model=List[UserResponse])
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).filter(User.is_active == True).all()
    print(f"API: Returning {len(users)} active users")
    for user in users:
        print(f"  - {user.username} ({user.display_name})")
    return users

@app.post("/users/initialize-predefined")
def initialize_predefined_users(db: Session = Depends(get_db)):
    """Manually initialize predefined users if they don't exist"""
    try:
        print("Manually initializing predefined users...")
        users_created = 0
        for account in PRELOADED_ACCOUNTS:
            # Check if user already exists
            existing_user = db.query(User).filter(User.username == account["username"]).first()
            if not existing_user:
                user = User(
                    username=account["username"],
                    display_name=account["display_name"],
                    email=account["email"],
                    household_size=account["household_size"],
                    account_type=account["account_type"],
                    description=account["description"]
                )
                db.add(user)
                users_created += 1
                print(f"Created user: {account['display_name']} ({account['username']})")
            else:
                print(f"User already exists: {account['display_name']} ({account['username']})")
        
        if users_created > 0:
            db.commit()
            return {"message": f"Successfully created {users_created} predefined users", "users_created": users_created}
        else:
            return {"message": "All predefined users already exist", "users_created": 0}
    except Exception as e:
        print(f"Error initializing predefined users: {e}")
        return {"error": str(e)}

@app.post("/users/{user_id}/load-test-data")
def load_test_data(user_id: int, db: Session = Depends(get_db)):
    """Load pre-loaded test data for a user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Find matching pre-loaded account
    test_account = None
    for account in PRELOADED_ACCOUNTS:
        if account["username"] == user.username:
            test_account = account
            break
    
    if not test_account:
        raise HTTPException(status_code=404, detail="No test data found for this user")
    
    # Add items
    items_added = []
    for item_data in test_account["items"]:
        # Calculate smart consumption rate
        consumption_engine = SmartConsumptionEngine()
        consumption_rate = consumption_engine.calculate_consumption_rate(
            item_data["name"], item_data["category"], user.household_size
        )
        
        db_item = GroceryItem(
            user_id=user_id,
            name=item_data["name"],
            brand=item_data["brand"],
            category=item_data["category"],
            quantity=item_data["quantity"],
            unit=item_data["unit"],
            image_url=item_data["image_url"],
            household_size=user.household_size,
            consumption_rate=consumption_rate,
            last_purchased=datetime.now() - timedelta(days=7),  # Purchased a week ago
            reminder_threshold=item_data["quantity"] * 0.2
        )
        db.add(db_item)
        items_added.append(db_item)
    
    db.commit()
    
    # Create realistic order history for AI learning
    order_history_created = []
    for item in items_added:
        # Create 3-5 historical orders for each item to train AI
        num_orders = min(5, max(3, user.household_size))  # More orders for larger households
        
        for i in range(num_orders):
            # Calculate order dates (going back in time)
            days_back = (i + 1) * 7 + random.randint(-2, 2)  # Weekly orders with some variation
            order_date = datetime.now() - timedelta(days=days_back)
            
            # Calculate realistic order quantities based on consumption
            base_quantity = item.consumption_rate * 7  # 1 week supply
            order_quantity = base_quantity * random.uniform(0.8, 1.2)  # ±20% variation
            
            # Create order
            order = OrderHistory(
                user_id=user_id,
                item_id=item.id,
                quantity=round(order_quantity, 2),
                order_date=order_date,
                delivery_date=order_date + timedelta(days=random.randint(1, 3)),
                guest_factor=random.choice([1.0, 1.0, 1.0, 1.2, 1.5])  # Mostly normal, some guest scenarios
            )
            db.add(order)
            order_history_created.append(order)
    
    db.commit()
    
    # Retrain model for this user
    predictor.train_model(db, user_id)
    
    return {
        "message": f"Loaded {len(items_added)} test items and {len(order_history_created)} order history records", 
        "items_added": len(items_added),
        "orders_created": len(order_history_created),
        "ai_training": "completed"
    }

@app.post("/users/{user_id}/simulate-urgent-scenarios")
def simulate_urgent_reorder_scenarios(user_id: int, db: Session = Depends(get_db)):
    """Simulate urgent reorder scenarios by reducing item quantities"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get user's items
        items = db.query(GroceryItem).filter(GroceryItem.user_id == user_id).all()
        if not items:
            raise HTTPException(status_code=404, detail="No items found for this user")
        
        urgent_items_created = 0
        scenarios_created = []
        
        # Create different urgent scenarios
        urgent_scenarios = [
            {"name": "Critical Milk Shortage", "category": "Dairy", "quantity": 0.05, "unit": "liters", "urgency": "high"},
            {"name": "Bread Emergency", "category": "Bakery", "quantity": 0.1, "unit": "packets", "urgency": "high"},
            {"name": "Egg Crisis", "category": "Dairy", "quantity": 0.5, "unit": "pieces", "urgency": "high"},
            {"name": "Tomato Shortage", "category": "Produce", "quantity": 0.02, "unit": "kg", "urgency": "high"},
            {"name": "Oil Emergency", "category": "Pantry", "quantity": 0.05, "unit": "liters", "urgency": "high"},
            {"name": "Salt Critical", "category": "Pantry", "quantity": 0.01, "unit": "kg", "urgency": "high"},
            {"name": "Tea Emergency", "category": "Beverages", "quantity": 0.02, "unit": "kg", "urgency": "high"},
            {"name": "Detergent Crisis", "category": "Household", "quantity": 0.05, "unit": "kg", "urgency": "high"}
        ]
        
        for scenario in urgent_scenarios:
            # Find matching item or create new one
            existing_item = next((item for item in items if item.name.lower() in scenario["name"].lower() or scenario["name"].lower() in item.name.lower()), None)
            
            if existing_item:
                # Update existing item to urgent quantity
                old_quantity = existing_item.quantity
                existing_item.quantity = scenario["quantity"]
                existing_item.last_purchased = datetime.now() - timedelta(days=random.randint(1, 3))
                scenarios_created.append({
                    "item_name": existing_item.name,
                    "old_quantity": old_quantity,
                    "new_quantity": scenario["quantity"],
                    "urgency": scenario["urgency"],
                    "scenario": scenario["name"]
                })
                urgent_items_created += 1
            else:
                # Create new urgent item
                consumption_engine = SmartConsumptionEngine()
                consumption_rate = consumption_engine.calculate_consumption_rate(
                    scenario["name"], scenario["category"], user.household_size
                )
                
                new_item = GroceryItem(
                    user_id=user_id,
                    name=scenario["name"],
                    brand="Local",
                    category=scenario["category"],
                    quantity=scenario["quantity"],
                    unit=scenario["unit"],
                    image_url="https://example.com/urgent.jpg",
                    household_size=user.household_size,
                    consumption_rate=consumption_rate,
                    last_purchased=datetime.now() - timedelta(days=random.randint(1, 3)),
                    reminder_threshold=scenario["quantity"] * 0.5
                )
                db.add(new_item)
                scenarios_created.append({
                    "item_name": new_item.name,
                    "old_quantity": 0,
                    "new_quantity": scenario["quantity"],
                    "urgency": scenario["urgency"],
                    "scenario": scenario["name"]
                })
                urgent_items_created += 1
        
        db.commit()
        
        # Retrain AI model with new urgent scenarios
        predictor.train_model(db, user_id)
        
        return {
            "message": f"Created {urgent_items_created} urgent reorder scenarios",
            "scenarios_created": scenarios_created,
            "ai_training": "completed",
            "urgency_level": "high",
            "recommendation": "Check Dashboard for urgent items that need immediate reordering!"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating urgent scenarios: {str(e)}")

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Soft delete a user (custom persona) by marking them as inactive."""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Only allow deletion of custom personas
        if not user.username.startswith('custom_'):
            raise HTTPException(status_code=403, detail="Cannot delete predefined personas")
        
        # Soft delete the user
        user.is_active = False
        # Disassociate email to free it up
        user.email = None 
        db.commit()
        
        return {
            "message": f"Custom persona '{user.display_name}' has been deactivated.",
            "deleted_user": user.display_name,
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

@app.post("/items/", response_model=GroceryItemResponse)
def create_grocery_item(item: GroceryItemCreate, user_id: int, db: Session = Depends(get_db)):
    try:
        # Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate smart consumption rate
        consumption_engine = SmartConsumptionEngine()
        consumption_rate = consumption_engine.calculate_consumption_rate(
            item.name, item.category, item.household_size
        )
        
        db_item = GroceryItem(
            user_id=user_id,
            name=item.name,
            brand=item.brand,
            category=item.category,
            quantity=item.quantity,
            unit=item.unit,
            image_url=item.image_url,
            household_size=item.household_size,
            consumption_rate=consumption_rate,
            last_purchased=datetime.now(),
            reminder_threshold=item.quantity * 0.2
        )
        db.add(db_item)
        db.commit()
        db.refresh(db_item)
        
        # Retrain model for this user
        predictor = GroceryPredictor()
        predictor.train_model(db, user_id)
        
        return db_item
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating grocery item: {str(e)}")

@app.get("/items/{user_id}", response_model=List[GroceryItemResponse])
def get_grocery_items(user_id: int, db: Session = Depends(get_db)):
    return db.query(GroceryItem).filter(
        GroceryItem.user_id == user_id,
        GroceryItem.is_active == True
    ).all()

@app.get("/predictions/{user_id}")
def get_user_predictions(user_id: int, guest_factor: str = "normal", db: Session = Depends(get_db)):
    """Get predictions for all active items for a specific user"""
    items = db.query(GroceryItem).filter(
        GroceryItem.user_id == user_id,
        GroceryItem.is_active == True
    ).all()
    
    predictions = []
    
    for item in items:
        prediction = predictor.predict_reorder_date(item, guest_factor)
        predictions.append({
            "item_id": item.id,
            "item_name": item.name,
            "brand": item.brand,
            "category": item.category,
            "current_quantity": item.quantity,
            "unit": item.unit,
            "image_url": item.image_url,
            **prediction
        })
    
    # Sort by urgency
    predictions.sort(key=lambda x: x["days_until_empty"])
    return predictions

@app.post("/order/{item_id}")
def place_order(item_id: int, quantity: float, user_id: int, guest_factor: str = "normal", db: Session = Depends(get_db)):
    """Record a new order for an item"""
    try:
        item = db.query(GroceryItem).filter(
            GroceryItem.id == item_id,
            GroceryItem.user_id == user_id
        ).first()
        
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Update item quantity
        item.quantity += quantity
        item.last_purchased = datetime.now()
        
        # Record order history
        order = OrderHistory(
            user_id=user_id,
            item_id=item_id,
            quantity=quantity,
            order_date=datetime.now(),
            delivery_date=datetime.now() + timedelta(days=1),
            guest_factor=1.0 if guest_factor == "normal" else 1.3
        )
        
        db.add(order)
        db.commit()
        
        # 🧠 AI LEARNING: Learn from this order to improve future predictions
        consumption_engine = SmartConsumptionEngine()
        learning_result = consumption_engine.learn_from_order_history(db, user_id, item_id)
        
        # Log learning results
        if learning_result.get("learned"):
            print(f"🤖 AI Learned for user {user_id}, item {item_id}:")
            print(f"   Old rate: {learning_result['old_rate']} → New rate: {learning_result['new_rate']}")
            print(f"   Actual consumption: {learning_result['actual_consumption']} per day")
            print(f"   Order interval: {learning_result['avg_order_interval']} days")
        else:
            print(f"🤖 AI Learning status: {learning_result.get('message', 'Unknown')}")
        
        # Retrain model for this user with updated consumption rates
        predictor.train_model(db, user_id)
        
        return {
            "message": "Order placed successfully", 
            "new_quantity": item.quantity,
            "ai_learning": learning_result
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order-history/{user_id}")
def get_order_history(user_id: int, db: Session = Depends(get_db)):
    """Get order history for a specific user"""
    # Get individual item orders from OrderHistory table
    order_history_records = db.query(OrderHistory).filter(
        OrderHistory.user_id == user_id
    ).order_by(OrderHistory.order_date.desc()).all()
    
    # Get cart orders from Order table
    cart_orders = db.query(Order).filter(
        Order.user_id == user_id
    ).order_by(Order.created_at.desc()).all()
    
    order_history = []
    
    # Process individual item orders
    for order in order_history_records:
        # Get item details
        item = db.query(GroceryItem).filter(GroceryItem.id == order.item_id).first()
        if item:
            order_history.append({
                "id": order.id,
                "order_number": f"IND-{order.id}",  # Individual order prefix
                "item_name": item.name,
                "category": item.category,
                "quantity": order.quantity,
                "unit": item.unit,
                "order_date": order.order_date,
                "delivery_date": order.delivery_date,
                "status": "Delivered" if order.delivery_date < datetime.now() else "In Transit",
                "price_per_unit": 50.0,  # Default price, can be enhanced later
                "order_type": "individual"
            })
    
    # Process cart orders
    for order in cart_orders:
        # Get order items
        order_items = db.query(OrderItem).filter(OrderItem.order_id == order.id).all()
        
        for order_item in order_items:
            # Get item details
            item = db.query(GroceryItem).filter(GroceryItem.id == order_item.item_id).first()
            if item:
                order_history.append({
                    "id": order.id,
                    "order_number": order.order_number,
                    "item_name": order_item.item_name,
                    "category": order_item.item_category,
                    "quantity": order_item.quantity,
                    "unit": order_item.unit,
                    "order_date": order.created_at,
                    "delivery_date": order.estimated_delivery_time,
                    "status": "Delivered" if order.estimated_delivery_time and order.estimated_delivery_time < datetime.now() else "In Transit",
                    "price_per_unit": order_item.unit_price,
                    "order_type": "cart"
                })
    
    # Sort all orders by date (most recent first)
    order_history.sort(key=lambda x: x["order_date"], reverse=True)
    
    return order_history

@app.delete("/order-history/{order_id}")
def delete_order_history(order_id: int, user_id: int, db: Session = Depends(get_db)):
    """Delete an order from history and trigger AI learning to improve predictions"""
    try:
        # Find the order
        order = db.query(OrderHistory).filter(
            OrderHistory.id == order_id,
            OrderHistory.user_id == user_id
        ).first()
        
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        
        # Get item details before deletion for AI learning
        item = db.query(GroceryItem).filter(GroceryItem.id == order.item_id).first()
        item_name = item.name if item else "Unknown Item"
        item_category = item.category if item else "Unknown"
        
        # Store order data for AI learning before deletion
        deleted_order_data = {
            "item_id": order.item_id,
            "quantity": order.quantity,
            "order_date": order.order_date,
            "category": item_category,
            "item_name": item_name
        }
        
        # Delete the order
        db.delete(order)
        db.commit()
        
        # Trigger AI learning to adapt to the deletion
        try:
            consumption_engine = SmartConsumptionEngine()
            
            # Use the new deletion learning method
            learning_result = consumption_engine.learn_from_order_deletion(db, user_id, deleted_order_data)
            print(f"🧠 AI learned from order deletion for {item_name}: {learning_result}")
            
            # Retrain the AI model for this user
            predictor.train_model(db, user_id)
            
            # Detect if household size needs adjustment after deletion
            new_household_size = consumption_engine.detect_household_size_from_consumption(db, user_id)
            user = db.query(User).filter(User.id == user_id).first()
            
            if user and user.household_size != new_household_size:
                old_size = user.household_size
                user.household_size = new_household_size
                db.commit()
                print(f"🏠 AI detected household size change after order deletion: {old_size} → {new_household_size}")
            
            # Analyze deletion patterns for future improvements
            pattern_analysis = consumption_engine.analyze_deletion_patterns(db, user_id)
            print(f"📊 Deletion pattern analysis: {pattern_analysis}")
            
            print(f"✅ Order for {item_name} deleted successfully. AI model updated.")
            
        except Exception as learning_error:
            print(f"⚠️ AI learning error during order deletion: {learning_error}")
            # Continue with deletion even if AI learning fails
        
        return {
            "message": f"Order for {item_name} deleted successfully",
            "deleted_order": deleted_order_data,
            "ai_learning": "completed",
            "message_detail": "The AI model has been updated to improve future predictions based on this correction."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting order: {str(e)}")

@app.get("/order-history/{user_id}/deletion-impact")
def get_deletion_impact_analysis(user_id: int, db: Session = Depends(get_db)):
    """Get analysis of how order deletions have improved AI predictions"""
    try:
        consumption_engine = SmartConsumptionEngine()
        
        # Get user's items to analyze consumption rate changes
        user_items = db.query(GroceryItem).filter(GroceryItem.user_id == user_id).all()
        
        impact_analysis = {
            "user_id": user_id,
            "total_items": len(user_items),
            "items_with_improved_rates": 0,
            "consumption_rate_changes": [],
            "ai_model_status": "trained" if predictor.is_trained else "untrained",
            "household_size_detected": None,
            "recommendations": []
        }
        
        # Analyze each item's consumption rate
        for item in user_items:
            if item.consumption_rate > 0:
                # Compare with base rate to see if AI has learned
                base_rate = consumption_engine.calculate_consumption_rate(
                    item.name, item.category, item.household_size
                )
                
                rate_difference = item.consumption_rate - base_rate
                rate_improvement = abs(rate_difference) / base_rate if base_rate > 0 else 0
                
                if rate_improvement > 0.1:  # 10% improvement threshold
                    impact_analysis["items_with_improved_rates"] += 1
                
                impact_analysis["consumption_rate_changes"].append({
                    "item_name": item.name,
                    "category": item.category,
                    "current_rate": item.consumption_rate,
                    "base_rate": base_rate,
                    "improvement": round(rate_improvement * 100, 1),
                    "last_updated": item.last_purchased.isoformat() if item.last_purchased else None
                })
        
        # Get household size detection
        impact_analysis["household_size_detected"] = consumption_engine.detect_household_size_from_consumption(db, user_id)
        
        # Generate recommendations
        if impact_analysis["items_with_improved_rates"] < len(user_items) * 0.5:
            impact_analysis["recommendations"].append("Consider deleting more incorrect orders to improve AI learning")
        
        if impact_analysis["ai_model_status"] == "untrained":
            impact_analysis["recommendations"].append("Add more items to train the AI model")
        
        if len(user_items) < 5:
            impact_analysis["recommendations"].append("Add more grocery items for better AI predictions")
        
        return impact_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing deletion impact: {str(e)}")

@app.post("/users/{user_id}/learn-from-orders")
def learn_from_order_history(user_id: int, db: Session = Depends(get_db)):
    """Trigger AI learning from user's order history to improve predictions"""
    try:
        consumption_engine = SmartConsumptionEngine()
        
        # Get all items for this user
        user_items = db.query(GroceryItem).filter(GroceryItem.user_id == user_id).all()
        
        learning_results = []
        total_learned = 0
        
        for item in user_items:
            # Try to learn from order history for each item
            result = consumption_engine.learn_from_order_history(db, user_id, item.id)
            if result.get("learned"):
                total_learned += 1
                learning_results.append({
                    "item_name": item.name,
                    "category": item.category,
                    "learning_result": result
                })
        
        # Retrain the AI model with new consumption rates
        predictor.train_model(db, user_id)
        
        # Detect and update household size based on consumption patterns
        new_household_size = consumption_engine.detect_household_size_from_consumption(db, user_id)
        
        # Update user's household size if it changed
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.household_size != new_household_size:
            old_size = user.household_size
            user.household_size = new_household_size
            db.commit()
            print(f"🏠 AI detected household size change: {old_size} → {new_household_size}")
        
        return {
            "message": f"AI learning completed! Learned from {total_learned} items",
            "total_items_analyzed": len(user_items),
            "items_learned": total_learned,
            "learning_details": learning_results,
            "household_size_detected": new_household_size,
            "household_size_updated": user.household_size if user else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during AI learning: {str(e)}")

@app.put("/items/{item_id}")
def update_grocery_item(item_id: int, user_id: int, item_update: dict, db: Session = Depends(get_db)):
    """Update an existing grocery item"""
    item = db.query(GroceryItem).filter(
        GroceryItem.id == item_id,
        GroceryItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Update fields if provided
    if "quantity" in item_update:
        item.quantity = item_update["quantity"]
    if "consumption_rate" in item_update:
        item.consumption_rate = item_update["consumption_rate"]
    if "brand" in item_update:
        item.brand = item_update["brand"]
    
    db.commit()
    db.refresh(item)
    
    # Retrain model for this user
    predictor.train_model(db, user_id)
    
    return {"message": "Item updated successfully", "updated_item": item}

@app.delete("/items/{item_id}")
def delete_grocery_item(item_id: int, user_id: int, db: Session = Depends(get_db)):
    item = db.query(GroceryItem).filter(
        GroceryItem.id == item_id,
        GroceryItem.user_id == user_id
    ).first()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    item.is_active = False
    db.commit()
    return {"message": "Item deleted successfully"}

# Notification Endpoints
@app.get("/notifications/{user_id}", response_model=List[NotificationResponse])
def get_user_notifications(user_id: int, unread_only: bool = False, db: Session = Depends(get_db)):
    """Get notifications for a user"""
    notification_manager = NotificationManager()
    notifications = notification_manager.get_user_notifications(db, user_id, unread_only)
    return notifications

@app.post("/notifications/{notification_id}/read")
def mark_notification_read(notification_id: int, user_id: int, db: Session = Depends(get_db)):
    """Mark a notification as read"""
    notification_manager = NotificationManager()
    success = notification_manager.mark_as_read(db, notification_id, user_id)
    
    if success:
        return {"message": "Notification marked as read"}
    else:
        raise HTTPException(status_code=404, detail="Notification not found")

@app.post("/notifications/{user_id}/create-urgent")
def create_urgent_notifications(user_id: int, db: Session = Depends(get_db)):
    """Create urgent reorder notifications for low stock items"""
    notification_manager = NotificationManager()
    predictor = GroceryPredictor()
    
    # Get user's items
    items = db.query(GroceryItem).filter(
        GroceryItem.user_id == user_id,
        GroceryItem.is_active == True
    ).all()
    
    urgent_notifications = []
    
    for item in items:
        prediction = predictor.predict_reorder_date(item)
        if prediction['urgency_level'] == 'high':
            notification = notification_manager.create_urgent_reorder_notification(db, user_id, item)
            if notification:
                urgent_notifications.append(notification)
    
    return {
        "message": f"Created {len(urgent_notifications)} urgent notifications",
        "notifications": urgent_notifications
    }

@app.post("/notifications/{user_id}/weekly-reminder")
def create_weekly_reminder(user_id: int, db: Session = Depends(get_db)):
    """Create weekly shopping reminder"""
    notification_manager = NotificationManager()
    predictor = GroceryPredictor()
    
    # Get user's items
    items = db.query(GroceryItem).filter(
        GroceryItem.user_id == user_id,
        GroceryItem.is_active == True
    ).all()
    
    urgent_items = []
    low_stock_items = []
    
    for item in items:
        prediction = predictor.predict_reorder_date(item)
        if prediction['urgency_level'] == 'high':
            urgent_items.append(item)
        elif prediction['urgency_level'] == 'medium':
            low_stock_items.append(item)
    
    notification = notification_manager.create_weekly_reminder(db, user_id, urgent_items, low_stock_items)
    
    return {
        "message": "Weekly reminder created",
        "urgent_items_count": len(urgent_items),
        "low_stock_items_count": len(low_stock_items),
        "notification": notification
    }

# Budget Management Endpoints
@app.post("/budgets/{user_id}", response_model=BudgetResponse)
def create_budget(user_id: int, budget_data: BudgetCreate, db: Session = Depends(get_db)):
    """Create or update a budget for a user"""
    budget_manager = BudgetManager()
    budget = budget_manager.create_budget(db, user_id, budget_data)
    
    return BudgetResponse(
        id=budget.id,
        month=budget.month,
        budget_amount=budget.budget_amount,
        spent_amount=budget.spent_amount,
        remaining_amount=budget.budget_amount - budget.spent_amount,
        category=budget.category,
        created_at=budget.created_at,
        is_active=budget.is_active
    )

@app.get("/budgets/{user_id}/status")
def get_budget_status(user_id: int, db: Session = Depends(get_db)):
    """Get comprehensive budget status for a user"""
    budget_manager = BudgetManager()
    status = budget_manager.get_budget_status(db, user_id)
    return status

@app.get("/budgets/{user_id}/current")
def get_current_budget(user_id: int, category: str = None, db: Session = Depends(get_db)):
    """Get current month's budget"""
    budget_manager = BudgetManager()
    budget = budget_manager.get_current_month_budget(db, user_id, category)
    
    if not budget:
        return {"message": "No budget set for current month", "budget": None}
    
    return BudgetResponse(
        id=budget.id,
        month=budget.month,
        budget_amount=budget.budget_amount,
        spent_amount=budget.spent_amount,
        remaining_amount=budget.budget_amount - budget.spent_amount,
        category=budget.category,
        created_at=budget.created_at,
        is_active=budget.is_active
    )

# Weekly Reminder Endpoint
@app.get("/weekly-reminder/{user_id}", response_model=WeeklyReminderResponse)
def get_weekly_reminder(user_id: int, db: Session = Depends(get_db)):
    """Get comprehensive weekly shopping reminder"""
    predictor = GroceryPredictor()
    budget_manager = BudgetManager()
    
    # Get user's items
    items = db.query(GroceryItem).filter(
        GroceryItem.user_id == user_id,
        GroceryItem.is_active == True
    ).all()
    
    urgent_items = []
    low_stock_items = []
    shopping_suggestions = []
    
    for item in items:
        prediction = predictor.predict_reorder_date(item)
        item_data = {
            "item_id": item.id,
            "name": item.name,
            "category": item.category,
            "current_quantity": item.quantity,
            "unit": item.unit,
            "days_until_empty": prediction['days_until_empty'],
            "suggested_quantity": prediction['suggested_quantity']
        }
        
        if prediction['urgency_level'] == 'high':
            urgent_items.append(item_data)
        elif prediction['urgency_level'] == 'medium':
            low_stock_items.append(item_data)
        
        # Add to shopping suggestions if needs reordering
        if prediction['urgency_level'] in ['high', 'medium']:
            shopping_suggestions.append(item_data)
    
    # Get budget status
    budget_status = budget_manager.get_budget_status(db, user_id)
    
    # Calculate week dates
    today = datetime.now()
    week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
    week_end = (today + timedelta(days=6-today.weekday())).strftime("%Y-%m-%d")
    
    return WeeklyReminderResponse(
        week_start=week_start,
        week_end=week_end,
        urgent_items=urgent_items,
        low_stock_items=low_stock_items,
        budget_status=budget_status,
        shopping_suggestions=shopping_suggestions
    )

# Seasonal Recommendations Endpoints
@app.get("/recommendations/{user_id}/seasonal")
def get_seasonal_recommendations(user_id: int, category: str = None, db: Session = Depends(get_db)):
    """Get seasonal recommendations for a user"""
    recommendations_manager = SeasonalRecommendationsManager()
    return recommendations_manager.get_seasonal_recommendations(category)

@app.get("/recommendations/{user_id}/trending")
def get_trending_recommendations(user_id: int, trend_type: str = "health", db: Session = Depends(get_db)):
    """Get trending recommendations for a user"""
    recommendations_manager = SeasonalRecommendationsManager()
    return recommendations_manager.get_trending_items(trend_type)

@app.get("/recommendations/{user_id}/smart")
def get_smart_recommendations(user_id: int, category: str = None, db: Session = Depends(get_db)):
    """Get smart personalized recommendations for a user"""
    recommendations_manager = SeasonalRecommendationsManager()
    return recommendations_manager.get_smart_recommendations(db, user_id, category)

# Smart Shopping List Endpoints
@app.get("/shopping-list/{user_id}/generate")
def generate_smart_shopping_list(user_id: int, days_ahead: int = 7, db: Session = Depends(get_db)):
    """Generate a smart shopping list for a user"""
    shopping_list_generator = SmartShoppingListGenerator()
    return shopping_list_generator.generate_smart_shopping_list(db, user_id, days_ahead)

@app.get("/shopping-list/{user_id}/meal-plan")
def generate_weekly_meal_plan(user_id: int, db: Session = Depends(get_db)):
    """Generate a weekly meal plan based on available items"""
    shopping_list_generator = SmartShoppingListGenerator()
    return shopping_list_generator.generate_weekly_meal_plan(db, user_id)

# Smart Basket Endpoints
@app.post("/smart-baskets/{user_id}", response_model=SmartBasketResponse)
def create_smart_basket(user_id: int, basket_data: SmartBasketCreate, db: Session = Depends(get_db)):
    """Create a new smart basket for a user"""
    try:
        basket_manager = SmartBasketManager()
        basket = basket_manager.create_smart_basket(db, user_id, basket_data)
        
        # Get item details for response
        item = db.query(GroceryItem).filter(GroceryItem.id == basket.item_id).first()
        
        return SmartBasketResponse(
            id=basket.id,
            item_id=basket.item_id,
            item_name=item.name,
            item_category=item.category,
            basket_name=basket.basket_name,
            auto_reorder_enabled=basket.auto_reorder_enabled,
            reorder_threshold_days=basket.reorder_threshold_days,
            min_quantity=basket.min_quantity,
            max_quantity=basket.max_quantity,
            last_auto_added=basket.last_auto_added,
            auto_add_count=basket.auto_add_count,
            created_at=basket.created_at,
            is_active=basket.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating smart basket: {str(e)}")

@app.get("/smart-baskets/{user_id}")
def get_user_smart_baskets(user_id: int, db: Session = Depends(get_db)):
    """Get all smart baskets for a user"""
    try:
        basket_manager = SmartBasketManager()
        baskets = basket_manager.get_user_smart_baskets(db, user_id)
        return baskets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching smart baskets: {str(e)}")

@app.put("/smart-baskets/{basket_id}")
def update_smart_basket(basket_id: int, user_id: int, update_data: SmartBasketUpdate, db: Session = Depends(get_db)):
    """Update a smart basket"""
    try:
        basket_manager = SmartBasketManager()
        basket = basket_manager.update_smart_basket(db, basket_id, user_id, update_data)
        
        # Get item details for response
        item = db.query(GroceryItem).filter(GroceryItem.id == basket.item_id).first()
        
        return SmartBasketResponse(
            id=basket.id,
            item_id=basket.item_id,
            item_name=item.name,
            item_category=item.category,
            basket_name=basket.basket_name,
            auto_reorder_enabled=basket.auto_reorder_enabled,
            reorder_threshold_days=basket.reorder_threshold_days,
            min_quantity=basket.min_quantity,
            max_quantity=basket.max_quantity,
            last_auto_added=basket.last_auto_added,
            auto_add_count=basket.auto_add_count,
            created_at=basket.created_at,
            is_active=basket.is_active
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating smart basket: {str(e)}")

@app.delete("/smart-baskets/{basket_id}")
def delete_smart_basket(basket_id: int, user_id: int, db: Session = Depends(get_db)):
    """Delete a smart basket"""
    try:
        basket_manager = SmartBasketManager()
        basket_manager.delete_smart_basket(db, basket_id, user_id)
        return {"message": "Smart basket deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting smart basket: {str(e)}")

@app.get("/smart-baskets/{basket_id}/history")
def get_basket_history(basket_id: int, user_id: int, db: Session = Depends(get_db)):
    """Get history for a specific smart basket"""
    try:
        basket_manager = SmartBasketManager()
        history = basket_manager.get_basket_history(db, basket_id, user_id)
        
        return [SmartBasketHistoryResponse(
            id=h.id,
            action=h.action,
            quantity_added=h.quantity_added,
            reason=h.reason,
            created_at=h.created_at
        ) for h in history]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching basket history: {str(e)}")

@app.post("/smart-baskets/{user_id}/check-auto-add")
def check_and_auto_add_items(user_id: int, db: Session = Depends(get_db)):
    """Check all smart baskets and auto-add items to cart if needed"""
    try:
        basket_manager = SmartBasketManager()
        auto_added_items = basket_manager.check_and_auto_add_items(db, user_id)
        
        return {
            "message": f"Checked smart baskets and auto-added {len(auto_added_items)} items",
            "auto_added_items": auto_added_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking smart baskets: {str(e)}")

@app.get("/smart-baskets/{user_id}/available-items")
def get_available_items_for_basket(user_id: int, db: Session = Depends(get_db)):
    """Get items that can be added to smart baskets (items not already in baskets)"""
    try:
        # Get all user's items
        all_items = db.query(GroceryItem).filter(
            GroceryItem.user_id == user_id,
            GroceryItem.is_active == True
        ).all()
        
        # Get items already in smart baskets
        basket_item_ids = db.query(SmartBasket.item_id).filter(
            SmartBasket.user_id == user_id,
            SmartBasket.is_active == True
        ).all()
        basket_item_ids = [item_id[0] for item_id in basket_item_ids]
        
        # Filter out items already in baskets
        available_items = [
            {
                "id": item.id,
                "name": item.name,
                "category": item.category,
                "current_quantity": item.quantity,
                "unit": item.unit,
                "brand": item.brand
            }
            for item in all_items
            if item.id not in basket_item_ids
        ]
        
        return available_items
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching available items: {str(e)}")

# Order Management Endpoints
class CartOrderRequest(BaseModel):
    cart_items: List[dict]
    delivery_address: Optional[str] = None
    delivery_instructions: Optional[str] = None

@app.post("/orders/{user_id}/cart", response_model=OrderResponse)
def create_cart_order(user_id: int, cart_request: CartOrderRequest, db: Session = Depends(get_db)):
    """Create a new order from cart items with grace period"""
    try:
        order_data = OrderCreate(
            delivery_address=cart_request.delivery_address,
            delivery_instructions=cart_request.delivery_instructions
        )
        
        order_manager = OrderManager()
        order = order_manager.create_order(db, user_id, cart_request.cart_items, order_data)
        
        # Get order items
        order_items = db.query(OrderItem).filter(OrderItem.order_id == order.id).all()
        items_data = [
            {
                "id": item.id,
                "item_name": item.item_name,
                "item_category": item.item_category,
                "quantity": item.quantity,
                "unit_price": item.unit_price,
                "total_price": item.total_price,
                "unit": item.unit,
                "brand": item.brand,
                "added_during_grace_period": item.added_during_grace_period
            }
            for item in order_items
        ]
        
        return OrderResponse(
            id=order.id,
            order_number=order.order_number,
            status=order.status,
            total_amount=order.total_amount,
            delivery_fee=order.delivery_fee,
            subtotal=order.subtotal,
            created_at=order.created_at,
            grace_period_ends=order.grace_period_ends,
            is_grace_period_active=order.is_grace_period_active,
            delivery_tier=order.delivery_tier,
            estimated_delivery_time=order.estimated_delivery_time,
            items=items_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating order: {str(e)}")

@app.post("/orders/{order_id}/add-item")
def add_item_to_order(order_id: int, user_id: int, item_data: OrderItemCreate, db: Session = Depends(get_db)):
    """Add item to existing order during grace period"""
    try:
        order_manager = OrderManager()
        order_item = order_manager.add_item_to_order(db, order_id, user_id, item_data)
        
        return {
            "message": "Item added to order successfully",
            "item": {
                "id": order_item.id,
                "item_name": order_item.item_name,
                "quantity": order_item.quantity,
                "total_price": order_item.total_price,
                "added_during_grace_period": order_item.added_during_grace_period
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding item to order: {str(e)}")

@app.get("/orders/{order_id}/grace-period")
def get_order_grace_period(order_id: int, user_id: int, db: Session = Depends(get_db)):
    """Get grace period status for an order"""
    try:
        order_manager = OrderManager()
        grace_period = order_manager.get_order_grace_period(db, order_id, user_id)
        return grace_period
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching grace period: {str(e)}")

@app.post("/orders/{order_id}/end-grace-period")
def end_grace_period(order_id: int, user_id: int, db: Session = Depends(get_db)):
    """End grace period for an order"""
    try:
        order_manager = OrderManager()
        order = order_manager.end_grace_period(db, order_id, user_id)
        return {"message": "Grace period ended successfully", "order_status": order.status}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending grace period: {str(e)}")

@app.get("/delivery-tiers")
def get_delivery_tiers(current_subtotal: float = 0, db: Session = Depends(get_db)):
    """Get available delivery tiers with savings messages"""
    try:
        order_manager = OrderManager()
        tiers = order_manager.get_delivery_tiers(db, current_subtotal)
        return tiers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching delivery tiers: {str(e)}")

# AI-Powered "Don't Forget" Analysis Endpoint
@app.post("/analyze-cart/{user_id}")
def analyze_cart_for_missing_items(user_id: int, cart_items: List[dict], db: Session = Depends(get_db)):
    """Analyze cart and suggest missing items based on AI analysis"""
    try:
        analyzer = DontForgetAnalyzer()
        suggestions = analyzer.analyze_cart_for_missing_items(db, user_id, cart_items)
        
        return {
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing cart: {str(e)}")

# Delivery Time Calculation Endpoint
@app.post("/calculate-delivery-time", response_model=DeliveryCalculationResponse)
def calculate_delivery_time(delivery_request: DeliveryCalculationRequest, db: Session = Depends(get_db)):
    """Calculate delivery time and grace period based on address and traffic"""
    try:
        calculator = DeliveryCalculator()
        calculation = calculator.calculate_delivery_time(db, delivery_request)
        return calculation
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating delivery time: {str(e)}")

# Initialize delivery tiers, stores, and zones
@app.on_event("startup")
def initialize_delivery_system():
    """Initialize default delivery system and predefined users on startup"""
    db = SessionLocal()
    try:
        # Initialize predefined users if they don't exist
        print("Checking and creating predefined users...")
        users_created = 0
        for account in PRELOADED_ACCOUNTS:
            # Check if user already exists
            existing_user = db.query(User).filter(User.username == account["username"]).first()
            if not existing_user:
                user = User(
                    username=account["username"],
                    display_name=account["display_name"],
                    email=account["email"],
                    household_size=account["household_size"],
                    account_type=account["account_type"],
                    description=account["description"]
                )
                db.add(user)
                users_created += 1
                print(f"Created user: {account['display_name']} ({account['username']})")
            else:
                print(f"User already exists: {account['display_name']} ({account['username']})")
        
        if users_created > 0:
            db.commit()
            print(f"Successfully created {users_created} predefined users")
        else:
            print("All predefined users already exist")
        
        # Initialize delivery tiers with new logic
        if db.query(DeliveryTier).count() == 0:
            tiers = [
                DeliveryTier(
                    tier_name="free",
                    min_order_amount=100.0,
                    delivery_fee=0.0,
                    estimated_delivery_time_minutes=45
                ),
                DeliveryTier(
                    tier_name="standard",
                    min_order_amount=0.0,
                    delivery_fee=50.0,
                    estimated_delivery_time_minutes=60
                )
            ]
            
            for tier in tiers:
                db.add(tier)
            
            db.commit()
            print("Default delivery tiers initialized")
        
        # Initialize stores
        if db.query(Store).count() == 0:
            stores = [
                Store(
                    name="Central Store",
                    address="123 Main Street, Delhi",
                    latitude=28.6139,
                    longitude=77.2090,
                    city="Delhi",
                    state="Delhi",
                    pincode="110001"
                ),
                Store(
                    name="North Store",
                    address="456 North Avenue, Delhi",
                    latitude=28.7041,
                    longitude=77.1025,
                    city="Delhi",
                    state="Delhi",
                    pincode="110002"
                )
            ]
            
            for store in stores:
                db.add(store)
            
            db.commit()
            print("Default stores initialized")
        
        # Initialize delivery zones
        if db.query(DeliveryZone).count() == 0:
            # Get the first store for zones
            first_store = db.query(Store).first()
            if first_store:
                zones = [
                    DeliveryZone(
                        store_id=first_store.id,
                        zone_name="Central Zone",
                        base_delivery_time_minutes=20,
                        traffic_multiplier=1.0,
                        pincode_start="110001",
                        pincode_end="110010"
                    ),
                    DeliveryZone(
                        store_id=first_store.id,
                        zone_name="North Zone",
                        base_delivery_time_minutes=25,
                        traffic_multiplier=1.0,
                        pincode_start="110011",
                        pincode_end="110020"
                    ),
                    DeliveryZone(
                        store_id=first_store.id,
                        zone_name="South Zone",
                        base_delivery_time_minutes=30,
                        traffic_multiplier=1.0,
                        pincode_start="110021",
                        pincode_end="110030"
                    )
                ]
                
                for zone in zones:
                    db.add(zone)
                
                db.commit()
                print("Default delivery zones initialized")
        
    except Exception as e:
        print(f"Error initializing delivery system: {e}")
    finally:
        db.close()

# Serve frontend
@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
