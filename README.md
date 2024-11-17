# Hierarchical Level Image Classification

## Project Overview
This project aims to classify images into bi-level categories, where each image belongs to a main category and a corresponding sub-category (e.g., IAB categories). The goal is to leverage deep learning models and computer vision techniques to effectively classify both the main category and the sub-category for each input image.

## IAB Categories
1. IAB1 - Arts & Entertainment
- IAB1-1 - Books & Literature
- IAB1-2 - Celebrity Fan/Gossip
- IAB1-3 - Fine Art
- IAB1-4 - Humor
- IAB1-5 - Movies
- IAB1-6 - Music 
- IAB1-7 - Television
2. IAB2 - Automotive
- IAB2-1 - Auto Parts
- IAB2-2 - Auto Repair
- IAB2-3 - Buying/Selling Cars
- IAB2-4 - Car Culture
- IAB2-5 - Certified Pre-Owned
- IAB2-6 - Convertible
- IAB2-7 - Coupe
- IAB2-8 - Crossover
- IAB2-9 - Diesel
- IAB2-10 - Electric Vehicle
- IAB2-11 - Hatchback
- IAB2-12 - Hybrid
- IAB2-13 - Luxury
- IAB2-14 - Minivan
- IAB2-15 - Motorcycles
- IAB2-16 - Off-Road Vehicles
- IAB2-17 - Performance Vehicles
- IAB2-18 - Pickup
- IAB2-19 - Road-Side Assistance
- IAB2-20 - Sedan
- IAB2-21 - Trucks & Accessories
- IAB2-22 - Vintage Cars
- IAB2-23 - Wagon
3. IAB3 - Business
- IAB3-1 - Advertising
- IAB3-2 - Agriculture
- IAB3-3 - Biotech/Biomedical
- IAB3-4 - Business Software
- IAB3-5 - Construction
- IAB3-6 - Forestry
- IAB3-7 - Government
- IAB3-8 - Green Solutions
- IAB3-9 - Human Resources
- IAB3-10 - Logistics
- IAB3-11 - Marketing
- IAB3-12 - Metals
4. IAB4 - Careers
- IAB4-1 - Career Planning
- IAB4-2 - College
- IAB4-3 - Financial Aid
- IAB4-4 - Job Fairs
- IAB4-5 - Job Search
- IAB4-6 - Resume Writing/Advice
- IAB4-7 - Nursing
- IAB4-8 - Scholarships
- IAB4-9 - Telecommuting
- IAB4-10 - Military
- IAB4-11 - Career Advice
5. IAB5 - Education
- IAB5-1 - 7-12 Education
- IAB5-2 - Adult Education
- IAB5-3 - Art History
- IAB5-4 - College Administration
- IAB5-5 - College Life
- IAB5-6 - Distance Learning
- IAB5-7 - English as a 2nd Language
- IAB5-8 - Language Learning
- IAB5-9 - Graduate School 
- IAB5-10 - Homeschooling
- IAB5-11 - Homework/Study Tips
- IAB5-12 - K-6 Education 
- IAB5-13 - Private School
- IAB5-14 - Special Education
- IAB5-15 - Studying Business
6. IAB6 - Family & Parenting
- IAB6-1 - Adoption
- IAB6-2 - Babies & Toddlers
- IAB6-3 - Daycare/Pre School
- IAB6-4 - Family Internet
- IAB6-5 - Parenting - K-6 Kids
- IAB6-6 - Parenting teens
- IAB6-7 - Pregnancy
- IAB6-8 - Special Needs Kids
- IAB6-9 - Eldercare
7. IAB7 - Health & Fitness
- IAB7-1 - Exercise 
- IAB7-2 - ADD
- IAB7-3 - AIDS / HIV
- IAB7-4 - Allergies
- IAB7-5 - Alternative Medicine 
- IAB7-6 - Arthritis
- IAB7-7 - Asthma
- IAB7-8 - Autism / PDD / Asperger’s Syndrome 1
- IAB7-9 - Bipolar Disorder
- IAB7-10 - Brain Tumor
- IAB7-11 - Cancer
- IAB7-12 - Cholesterol
- IAB7-13 - Chronic Fatigue Syndrome
- IAB7-14 - Chronic Pain
- IAB7-15 - Cold & Flu
- IAB7-16 - Deafness
- IAB7-17 - Dental Care
- IAB7-18 - Depression
- IAB7-19 - Dermatology
- IAB7-20 - Diabetes
- IAB7-21 - Epilepsy
- IAB7-22 - GERD / Acid Reflux
- IAB7-23 - Headaches / Migraines 
- IAB7-24 - Heart Disease 
- IAB7-25 - Herbs for Health 
- IAB7-26 - Holistic Healing
- IAB7-27 - IBS / Crohn’s 
- IAB7-28 - Incest / Abuse Support
- IAB7-29 - Incontinence
- IAB7-30 - Infertility
- IAB7-31 - Men’s Health
- IAB7-32 - Nutrition
- IAB7-33 - Orthopedics
- IAB7-34 - Panic / Anxiety Disorders
- IAB7-35 - Pediatrics
- IAB7-36 - Physical Therapy
- IAB7-37 - Psychology / Psychiatry 
- IAB7-38 - Senior Health
- IAB7-39 - Sexuality
- IAB7-40 - Sleep Disorders
- IAB7-41 - Smoking Cessation
- IAB7-42 - Substance Abuse
- IAB7-43 - Thyroid Disease
- IAB7-44 - Weight Loss 
- IAB7-45 - Women’s Health
8. IAB8 - Food & Drink
- IAB8-1 - American Cuisine
- IAB8-2 - Barbecues & Grilling
- IAB8-3 - Cajun / Creole
- IAB8-4 - Chinese Cuisine
- IAB8-5 - Cocktails / Beer
- IAB8-6 - Coffee / Tea
- IAB8-7 - Cuisine-Specific
- IAB8-8 - Desserts & Baking
- IAB8-9 - Dining Out
- IAB8-10 - Food Allergies
- IAB8-11 - French Cuisine
- IAB8-12 - Health / Low-Fat Cooking
- IAB8-13 - Italian Cuisine
- IAB8-14 - Japanese Cuisine
- IAB8-15 - Mexican Cuisine
- IAB8-16 - Vegan
- IAB8-17 - Vegetarian
- IAB8-18 - Wine
9. IAB9 - Hobbies & Interests
- IAB9-1 - Art / Technology
- IAB9-2 - Arts & Crafts
- IAB9-3 - Beadwork
- IAB9-4 - Bird-Watching
- IAB9-5 - Board Games / Puzzles
- IAB9-6 - Candle & Soap Making
- IAB9-7 - Card Games
- IAB9-8 - Chess
- IAB9-9 - Cigars 
- IAB9-10 - Collecting
- IAB9-11 - Comic Books 
- IAB9-12 - Drawing / Sketching
- IAB9-13 - Freelance Writing 
- IAB9-14 - Genealogy
- IAB9-15 - Getting Published 
- IAB9-16 - Guitar 
- IAB9-17 - Home Recording
- IAB9-18 - Investors & Patents 
- IAB9-19 - Jewelry Making
- IAB9-20 - Magic & Illusion
- IAB9-21 - Needlework
- IAB9-22 - Painting
- IAB9-23 - Photography
- IAB9-24 - Radio
- IAB9-25 - Roleplaying Games
- IAB9-26 - Sci-Fi & Fantasy
- IAB9-27 - Scrapbooking
- IAB9-28 - Screenwriting
- IAB9-29 - Stamps & Coins
- IAB9-30 - Video & Computer Games
- IAB9-31 - Woodworking
10. IAB10 - Home & Garden
- IAB10-1 - Appliances
- IAB10-2 - Entertaining
- IAB10-3 - Environmental Safety
- IAB10-4 - Gardening
- IAB10-5 - Home Repair
- IAB10-6 - Home Theater
- IAB10-7 - Interior Decorating
- IAB10-8 - Landscaping
- IAB10-9 - Remodeling & Construction
11. IAB11 - Law, Government, & Politics
- IAB11-1 - Immigration
- IAB11-2 - Legal Issues
- IAB11-3 - Government Resources 1
- IAB11-4 - Politics
- IAB11-5 - Commentary
12. IAB12 - News
- IAB12-1 - International News
- IAB12-2 - National News
- IAB12-3 - Local News
13. IAB13 - Personal Finance
- IAB13-1 - Beginning Investing
- IAB13-2 - Credit / Debit & Loans
- IAB13-3 - Financial News
- IAB13-4 - Financial Planning
- IAB13-5 - Hedge Fund
- IAB13-6 - Insurance
- IAB13-7 - Investing
- IAB13-8 - Mutual Funds 
- IAB13-9 - Options
- IAB13-10 - Retirement Planning
- IAB13-11 - Stocks
- IAB13-12 - Tax Planning
14. IAB14 - Society
- IAB14-1 - Dating 
- IAB14-2 - Divorce Support
- IAB14-3 - Gay Life
- IAB14-4 - Marriage
- IAB14-5 - Senior Living
- IAB14-6 - Teens
- IAB14-7 - Weddings
- IAB14-8 - Ethnic Specific
15. IAB15 - Science
- IAB15-1 - Astrology
- IAB15-2 - Biology
- IAB15-3 - Chemistry
- IAB15-4 - Geology
- IAB15-5 - Paranormal Phenomena
- IAB15-6 - Physics
- IAB15-7 - Space / Astronomy
- IAB15-8 - Geography
- IAB15-9 - Botany
- IAB15-10 - Weather
16. IAB16 - Pets
- IAB16-1 - Aquariums
- IAB16-2 - Birds
- IAB16-3 - Cats
- IAB16-4 - Dogs
- IAB16-5 - Large Animals
- IAB16-6 - Reptiles
- IAB16-7 - Veterinary Medicine
17. IAB17 - Sports
- IAB17-1 - Auto Racing
IAB17-2 - Baseball 
IAB17-3 - Bicycling
IAB17-4 - Bodybuilding
IAB17-5 - Boxing
IAB17-6 - Canoeing / Kayaking
IAB17-7 - Cheerleading
IAB17-8 - Climbing
IAB17-9 - Cricket
IAB17-10 - Figure Skating
IAB17-11 - Fly Fishing
IAB17-12 - American Football 1
IAB17-13 - Freshwater Fishing
IAB17-14 - Game & Fish
IAB17-15 - Golf
IAB17-16 - Horse Racing
IAB17-17 - Horses
IAB17-18 - Hunting / Shooting
IAB17-19 - Inline Skating
IAB17-20 - Martial Arts
IAB17-21 - Mountain Biking
IAB17-22 - NASCAR Racing
IAB17-23 - Olympics
IAB17-24 - Paintball
IAB17-25 - Power & Motorcycles
IAB17-26 - Basketball
IAB17-27 - Ice Hockey
IAB17-28 - Rodeo
IAB17-29 - Rugby
IAB17-30 - Running / Jogging
IAB17-31 - Sailing / Boating 1
IAB17-32 - Saltwater Fishing
IAB17-33 - Scuba Diving
IAB17-34 - Skateboarding
IAB17-35 - Skiing
IAB17-36 - Snowboarding
IAB17-37 - Surfing / Body-Boarding
IAB17-38 - Swimming
IAB17-39 - Table Tennis / Ping-Pong
IAB17-40 - Tennis
IAB17-41 - Volleyball
IAB17-43 - Waterski / Wakeboard
- IAB17-44 - World Football / Soccer 1
18. IAB18 - Style & Fashion
- IAB18-1 - Beauty
- IAB18-2 - Body Art
- IAB18-3 - Fashion
- IAB18-4 - Jewelry
- IAB18-5 - Clothing
- IAB18-6 - Accessories
19. IAB19 - Technology & Computing
- IAB19-1 - 3-D Graphics
IAB19-2 - Animation
IAB19-3 - Antivirus Software
IAB19-4 - C / C++
IAB19-5 - Cameras & Camcorders
IAB19-6 - Cell Phones
IAB19-7 - Computer Certification
IAB19-8 - Computer Networking
IAB19-9 - Computer Peripherals
IAB19-10 - Computer Reviews
IAB19-11 - Data Centers
IAB19-12 - Databases
IAB19-13 - Desktop Publishing
IAB19-14 - Desktop Video
IAB19-15 - Email / Chat / Messaging 2
IAB19-16 - Graphics Software
IAB19-17 - Home Video / DVD
IAB19-18 - Internet Technology
IAB19-19 - Java
IAB19-20 - JavaScript
IAB19-21 - Mac Support
IAB19-22 - MP3 / MIDI
IAB19-23 - Net Conferencing
IAB19-24 - Net for Beginners
IAB19-25 - Network Security
IAB19-26 - Palmtops / PDAs / Tablets 1
IAB19-27 - PC Support
IAB19-28 - Portable
IAB19-29 - Entertainment / Virtual & Augmented Reality 1
IAB19-30 - Shareware / Freeware / Open Source 1
IAB19-31 - Unix / Linux 1
IAB19-32 - Visual Basic
IAB19-33 - Web Clip Art
IAB19-34 - Web Design / HTML
IAB19-35 - Web Search
- IAB19-36 - Windows
20. IAB20 - Travel
- IAB20-1 - Adventure Travel
IAB20-2 - Africa
IAB20-3 - Air Travel
IAB20-4 - Australia & New Zealand
IAB20-5 - Bed & Breakfast
IAB20-6 - Budget Travel
IAB20-7 - Business Travel
IAB20-8 - By US Locale
IAB20-9 - Camping
IAB20-10 - Canada
IAB20-11 - Caribbean
IAB20-12 - Cruises
IAB20-13 - Eastern Europe
IAB20-14 - Europe
IAB20-15 - France
IAB20-16 - Greece
IAB20-17 - Honeymoons / Getaways
IAB20-18 - Hotels
IAB20-19 - Italy
IAB20-20 - Japan / China 2
IAB20-21 - Mexico & Central America
IAB20-22 - National Parks
IAB20-23 - South America
IAB20-24 - Spas
IAB20-25 - Theme Parks
IAB20-26 - Traveling with Kids
- IAB20-27 - United Kingdom
21. IAB21 - Real Estate
- IAB21-1 - Apartments
- IAB21-2 - Architects
- IAB21-3 - Buying / Selling Homes
22. IAB22 - Shopping
- IAB22-1 - Contests & Freebies
- IAB22-2 - Couponing
- IAB22-3 - Comparison
- IAB22-4 - Engines
23. IAB23 - Religion & Spirituality
- IAB23-1 - Alternative Religions
IAB23-2 - Atheism / Agnosticism
IAB23-3 - Buddhism
IAB23-4 - Catholicism
IAB23-5 - Christianity
IAB23-6 - Hinduism
IAB23-7 - Islam
IAB23-8 - Judaism
IAB23-9 - Latter-Day Saints
- IAB23-10 - Pagan / Wiccan
24. IAB24 - Uncategorized
25. IAB25 - Non-Standard Content
- IAB25-1 - Unmoderated UGC / Message Boards 
- IAB25-2 - Extreme Graphic / Explicit Violence
- IAB25-3 - Adult Content 1
- IAB25-4 - Profane Content
- IAB25-5 - Hate Content
- IAB25-6 - Under Construction
- IAB25-7 - Incentivized
26. IAB26 - Illegal Content
- IAB26-1 - Illegal Content
- IAB26-2 - Warez
- IAB26-3 - Spyware / Malware / Malicious 
- IAB26-4 - Copyright Infringement

