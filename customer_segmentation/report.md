## Notes

- Keep dimenionality optimional at 90% after PCA REDUCTION
- keep N neigbours and min-samples to be D + 1 and features or dimensions after PCA must be around 1% to 5% of total rows in dataset
- eps will be decided after the graph
- but start min-samples from 5 then go to max 50 slowly and find the best result
- keep noise as low as possible like around 5%

## Identifying eps (Epsilon)

**_Epsilon (eps) in DBSCAN is the maximum distance between two data points for them to be considered part of the same neighborhood (i.e., potential members of the same cluster)._**

### Signs of Too Low eps

- Many points labeled as noise (very high percentage of noise, often above 30–40%).
- Very small clusters or just a few core points grouped, leaving many points outside.
- Cluster count is very low or zero—sometimes DBSCAN can’t form meaningful clusters at all.

### Signs of Too High eps

- Clusters are very large (can even merge all points into one giant cluster).
- Few/no points marked as noise (nearly all are assigned somewhere).
- Cluster boundaries are unclear—different groups blend into each other, losing segmentation meaning

## Observation 1 : WITH ALL COLUMNS EXCEPT "Customer ID", "Item Purchased", "Shipping Type"

BEFORE ENCODING COLUMNS

```
 #   Column                  Non-Null Count  Dtype
---  ------                  --------------  -----
 0   Age                     3900 non-null   float64
 1   Gender                  3900 non-null   object
 2   Category                3900 non-null   object
 3   Purchase Amount (USD)   3900 non-null   float64
 4   Location                3900 non-null   object
 5   Size                    3900 non-null   object
 6   Color                   3900 non-null   object
 7   Season                  3900 non-null   object
 8   Review Rating           3900 non-null   float64
 9   Subscription Status     3900 non-null   object
 10  Discount Applied        3900 non-null   object
 11  Promo Code Used         3900 non-null   object
 12  Previous Purchases      3900 non-null   float64
 13  Payment Method          3900 non-null   object
 14  Frequency of Purchases  3900 non-null   float64
```

AFTER ENCODING COLUMNS

```
RangeIndex: 3900 entries, 0 to 3899
Data columns (total 94 columns):
 #   Column                      Non-Null Count  Dtype
---  ------                      --------------  -----
 0   Age                         3900 non-null   float64
 1   Purchase Amount (USD)       3900 non-null   float64
 2   Size                        3900 non-null   float64
 3   Review Rating               3900 non-null   float64
 4   Previous Purchases          3900 non-null   float64
 5   Frequency of Purchases      3900 non-null   float64
 6   Gender_Male                 3900 non-null   bool
 7   Category_Clothing           3900 non-null   bool
 8   Category_Footwear           3900 non-null   bool
 9   Category_Outerwear          3900 non-null   bool
 10  Location_Alaska             3900 non-null   bool
 11  Location_Arizona            3900 non-null   bool
 12  Location_Arkansas           3900 non-null   bool
 13  Location_California         3900 non-null   bool
 14  Location_Colorado           3900 non-null   bool
 15  Location_Connecticut        3900 non-null   bool
 16  Location_Delaware           3900 non-null   bool
 17  Location_Florida            3900 non-null   bool
 18  Location_Georgia            3900 non-null   bool
 19  Location_Hawaii             3900 non-null   bool
 20  Location_Idaho              3900 non-null   bool
 21  Location_Illinois           3900 non-null   bool
 22  Location_Indiana            3900 non-null   bool
 23  Location_Iowa               3900 non-null   bool
 24  Location_Kansas             3900 non-null   bool
 25  Location_Kentucky           3900 non-null   bool
 26  Location_Louisiana          3900 non-null   bool
 27  Location_Maine              3900 non-null   bool
 28  Location_Maryland           3900 non-null   bool
 29  Location_Massachusetts      3900 non-null   bool
 30  Location_Michigan           3900 non-null   bool
 31  Location_Minnesota          3900 non-null   bool
 32  Location_Mississippi        3900 non-null   bool
 33  Location_Missouri           3900 non-null   bool
 34  Location_Montana            3900 non-null   bool
 35  Location_Nebraska           3900 non-null   bool
 36  Location_Nevada             3900 non-null   bool
 37  Location_New Hampshire      3900 non-null   bool
 38  Location_New Jersey         3900 non-null   bool
 39  Location_New Mexico         3900 non-null   bool
 40  Location_New York           3900 non-null   bool
 41  Location_North Carolina     3900 non-null   bool
 42  Location_North Dakota       3900 non-null   bool
 43  Location_Ohio               3900 non-null   bool
 44  Location_Oklahoma           3900 non-null   bool
 45  Location_Oregon             3900 non-null   bool
 46  Location_Pennsylvania       3900 non-null   bool
 47  Location_Rhode Island       3900 non-null   bool
 48  Location_South Carolina     3900 non-null   bool
 49  Location_South Dakota       3900 non-null   bool
 50  Location_Tennessee          3900 non-null   bool
 51  Location_Texas              3900 non-null   bool
 52  Location_Utah               3900 non-null   bool
 53  Location_Vermont            3900 non-null   bool
 54  Location_Virginia           3900 non-null   bool
 55  Location_Washington         3900 non-null   bool
 56  Location_West Virginia      3900 non-null   bool
 57  Location_Wisconsin          3900 non-null   bool
 58  Location_Wyoming            3900 non-null   bool
 59  Color_Black                 3900 non-null   bool
 60  Color_Blue                  3900 non-null   bool
 61  Color_Brown                 3900 non-null   bool
 62  Color_Charcoal              3900 non-null   bool
 63  Color_Cyan                  3900 non-null   bool
 64  Color_Gold                  3900 non-null   bool
 65  Color_Gray                  3900 non-null   bool
 66  Color_Green                 3900 non-null   bool
 67  Color_Indigo                3900 non-null   bool
 68  Color_Lavender              3900 non-null   bool
 69  Color_Magenta               3900 non-null   bool
 70  Color_Maroon                3900 non-null   bool
 71  Color_Olive                 3900 non-null   bool
 72  Color_Orange                3900 non-null   bool
 73  Color_Peach                 3900 non-null   bool
 74  Color_Pink                  3900 non-null   bool
 75  Color_Purple                3900 non-null   bool
 76  Color_Red                   3900 non-null   bool
 77  Color_Silver                3900 non-null   bool
 78  Color_Teal                  3900 non-null   bool
 79  Color_Turquoise             3900 non-null   bool
 80  Color_Violet                3900 non-null   bool
 81  Color_White                 3900 non-null   bool
 82  Color_Yellow                3900 non-null   bool
 83  Season_Spring               3900 non-null   bool
 84  Season_Summer               3900 non-null   bool
 85  Season_Winter               3900 non-null   bool
 86  Subscription Status_Yes     3900 non-null   bool
 87  Promo Code Used_Yes         3900 non-null   bool
 88  Payment Method_Cash         3900 non-null   bool
 89  Payment Method_Credit Card  3900 non-null   bool
 90  Payment Method_Debit Card   3900 non-null   bool
 91  Payment Method_PayPal       3900 non-null   bool
 92  Payment Method_Venmo        3900 non-null   bool
 93  Discount Applied_Yes        3900 non-null   bool
dtypes: bool(88), float64(6)
memory usage: 518.1 KB

```

```python
# At PCA 0.9
[
 {'score': 0.209,
  'eps': 7.7,
  'min_samples': 37,
  'n_clusters': 30,
  'noise': 27.33},
  {'score': 0.271,
  'eps': 7.4,
  'min_samples': 46,
  'n_clusters': 20,
  'noise': 65.74},
  {'score': 0.221,
  'eps': 7.2,
  'min_samples': 48,
  'n_clusters': 9,
  'noise': 86.97},
  {'score': 0.206,
  'eps': 7.7,
  'min_samples': 30,
  'n_clusters': 31,
  'noise': 24.38},
  {'score': 0.212,
  'eps': 7.3,
  'min_samples': 9,
  'n_clusters': 40,
  'noise': 23.13},
  {'score': 0.201,
  'eps': 7.7,
  'min_samples': 29,
  'n_clusters': 31,
  'noise': 23.44}
]

# At PCA 0.8

[
    {'score': 0.206,
  'eps': 7.1,
  'min_samples': 28,
  'n_clusters': 32,
  'noise': 25.08}
]

# At PCA 0.7
{'score': 0.249,
  'eps': 6.4,
  'min_samples': 25,
  'n_clusters': 36,
  'noise': 40.69}

# AT PCA = 0.6
{'score': 0.205,
  'eps': 5.9,
  'min_samples': 30,
  'n_clusters': 15,
  'noise': 79.46}
```

## Observation 2 : WITH ALL COLUMNS EXCEPT "Customer ID", "Item Purchased", "Shipping Type" and Feature Engineering ( Location to Localtion_Region and Age to Age_Group, Color to Color_Group)

1. After Age to Age Group

```python
[{'score': 0.236,
  'eps': 7.6,
  'min_samples': 25,
  'n_clusters': 35,
  'noise': 26.15}]
```

2. After Location to Location_Region

```python
[{'score': 0.279,
  'eps': 4.9,
  'min_samples': 25,
  'n_clusters': 25,
  'noise': 19.69}]

```

## Observation 3 : WITH ALL COLUMNS EXCEPT "Customer ID", "Item Purchased", "Shipping Type" and Feature Engineering ( Location to Localtion_Region and Age to Age_Group )

1. AT 0.9 PCA

```python

[
  {'score': 0.279,
'eps': 4.9,
'min_samples': 25,
'n_clusters': 25,
'noise': 19.69},
{'score': 0.206,
  'eps': 5.0,
  'min_samples': 16,
  'n_clusters': 16,
  'noise': 5.46},      # Best Till now
{'score': 0.254,
  'eps': 4.9,
  'min_samples': 10,
  'n_clusters': 20,
  'noise': 4.46},
  {'score': 0.252,
  'eps': 4.9,
  'min_samples': 5,
  'n_clusters': 20,
  'noise': 2.03},
  {'score': 0.254,
  'eps': 4.9,
  'min_samples': 9,
  'n_clusters': 20,
  'noise': 3.92},
  {'score': 0.252,
  'eps': 4.9,
  'min_samples': 3,
  'n_clusters': 20,
  'noise': 1.56} # NEW BEST
]

```

2. AT 0.95 PCA

```python

 [{'score': 0.225,
  'eps': 5.2,
  'min_samples': 11,
  'n_clusters': 18,
  'noise': 3.51},
{'score': 0.224,
  'eps': 5.2,
  'min_samples': 10,
  'n_clusters': 18,
  'noise': 3.03},
  {'score': 0.212,
  'eps': 5.2,
  'min_samples': 5,
  'n_clusters': 17,
  'noise': 1.62},
  ]
```

3. At PCA = 0.93

```python
[
{
  'score': 0.226,
  'eps': 5.0,
  'min_samples': 4,
  'n_clusters': 18,
  'noise': 2.05
},
{
  'score': 0.225,
  'eps': 5.0,
  'min_samples': 3,
  'n_clusters': 18,
  'noise': 1.79
},
{
  'score': 0.213,
  'eps': 5.1,
  'min_samples': 4,
  'n_clusters': 17,
  'noise': 1.26
}
]
```

## Observation 4 : WITH ALL COLUMNS EXCEPT "Customer ID", "Item Purchased", "Shipping Type" and Feature Engineering ( Location to Localtion_Region and Age to Age_Group ) with K-Means
