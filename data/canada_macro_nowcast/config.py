from __future__ import annotations

STATCAN_REALTIME = [
    {
        "pid": "36100491",
        "table_id": "36-10-0491",
        "title": "Historical monthly GDP by industry",
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "36100431",
        "table_id": "36-10-0431",
        "title": "Historical quarterly expenditure GDP",
        "equals": {
            "Seasonal adjustment": ["Seasonally adjusted at annual rates"],
            "Prices": [
                "Chained (2017) dollars",
                "Current prices",
                "Chained (2017) dollars percentage change",
            ],
        },
    },
    {
        "pid": "18100259",
        "table_id": "18-10-0259",
        "title": "Historical CPI-common, CPI-median and CPI-trim",
    },
    {
        "pid": "14100331",
        "table_id": "14-10-0331",
        "title": "Historical SEPH employment and weekly earnings",
        "hierarchy": {"North American Industry Classification System": 3},
    },
    {
        "pid": "12100165",
        "table_id": "12-10-0165",
        "title": "Historical merchandise imports and exports",
        "equals": {
            "Basis": ["Balance of payments"],
            "Seasonal adjustment": ["Seasonally adjusted"],
        },
        "hierarchy": {"North American Product Classification System": 1},
    },
    {
        "pid": "16100014",
        "table_id": "16-10-0014",
        "title": "Historical real manufacturing sales, orders and inventories",
        "archived": True,
    },
    {
        "pid": "16100015",
        "table_id": "16-10-0015",
        "title": "Historical manufacturing capacity utilization",
        "archived": True,
    },
    {
        "pid": "16100118",
        "table_id": "16-10-0118",
        "title": "Historical manufacturing sales, inventories and orders",
        "archived": True,
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
            "Seasonal adjustment": ["Seasonally adjusted"],
        },
        "regex": {
            "North American Industry Classification System": (
                r"(?:industries|Manufacturing \[31-33\]|\[3\d{2}\])$"
            )
        },
    },
    {
        "pid": "20100005",
        "table_id": "20-10-0005",
        "title": "Historical wholesale sales, price and volume",
        "archived": True,
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "20100019",
        "table_id": "20-10-0019",
        "title": "Historical wholesale sales",
        "archived": True,
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
            "Seasonal adjustment": ["Seasonally adjusted"],
        },
        "regex": {
            "North American Industry Classification System": (
                r"(?:Wholesale trade|\[(?:41|4\d{2})\])$"
            )
        },
    },
    {
        "pid": "20100020",
        "table_id": "20-10-0020",
        "title": "Historical wholesale inventories",
        "archived": True,
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
        },
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "20100081",
        "table_id": "20-10-0081",
        "title": "Historical retail sales",
        "archived": True,
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
        },
    },
    {
        "pid": "20100082",
        "table_id": "20-10-0082",
        "title": "Historical retail sales, price and volume",
        "archived": True,
    },
    {
        "pid": "36100430",
        "table_id": "36-10-0430",
        "title": "Historical quarterly income GDP diagnostic",
        "optional": True,
        "equals": {"Seasonal adjustment": ["Seasonally adjusted at annual rates"]},
    },
    {
        "pid": "36100042",
        "table_id": "36-10-0042",
        "title": "Historical current-account diagnostic",
        "optional": True,
    },
]

STATCAN_CURRENT = [
    {
        "pid": "18100004",
        "table_id": "18-10-0004",
        "title": "All-items CPI and major components",
        "source_kind": "current_nonrevised_history",
        "equals": {"Geography": ["Canada"]},
        "hierarchy": {"Products and product groups": 1},
    },
    {
        "pid": "18100256",
        "table_id": "18-10-0256",
        "title": "Core inflation measures forward snapshots",
        "source_kind": "forward_snapshot",
        "equals": {
            "Geography": ["Canada"],
            "Alternative measures": [
                "Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)",
                "Measure of core inflation based on a weighted median approach, CPI-median (year-over-year percent change)",
                "Measure of core inflation based on a trimmed mean approach, CPI-trim (year-over-year percent change)",
                "Measure of core inflation based on a weighted median approach, CPI-median (index, 198901=100)",
                "Measure of core inflation based on a trimmed mean approach, CPI-trim (index, 198901=100)",
            ],
        },
    },
    {
        "pid": "16100013",
        "table_id": "16-10-0013",
        "title": "Real manufacturing forward snapshots",
        "source_kind": "forward_snapshot",
    },
    {
        "pid": "16100012",
        "table_id": "16-10-0012",
        "title": "Manufacturing capacity utilization forward snapshots",
        "source_kind": "forward_snapshot",
    },
    {
        "pid": "16100047",
        "table_id": "16-10-0047",
        "title": "Manufacturing sales, inventories and orders forward snapshots",
        "source_kind": "forward_snapshot",
        "equals": {"Seasonal adjustment": ["Seasonally adjusted"]},
        "regex": {
            "North American Industry Classification System": (
                r"(?:industries|Manufacturing \[31-33\]|\[3\d{2}\])$"
            )
        },
    },
    {
        "pid": "20100003",
        "table_id": "20-10-0003",
        "title": "Wholesale price and volume forward snapshots",
        "source_kind": "forward_snapshot",
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "20100074",
        "table_id": "20-10-0074",
        "title": "Wholesale sales forward snapshots",
        "source_kind": "forward_snapshot",
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
        },
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "20100076",
        "table_id": "20-10-0076",
        "title": "Wholesale inventories forward snapshots",
        "source_kind": "forward_snapshot",
        "equals": {"Adjustments": ["Seasonally adjusted"]},
        "hierarchy": {"North American Industry Classification System": 1},
    },
    {
        "pid": "20100056",
        "table_id": "20-10-0056",
        "title": "Retail sales forward snapshots",
        "source_kind": "forward_snapshot",
        "equals": {
            "Geography": ["Canada"],
            "Adjustments": ["Seasonally adjusted"],
        },
    },
    {
        "pid": "20100067",
        "table_id": "20-10-0067",
        "title": "Retail price and volume forward snapshots",
        "source_kind": "forward_snapshot",
    },
]

VALET_TARGETS = {
    "target_overnight": {
        "known_id": "V39079",
        "label_terms": ("target for the overnight rate",),
    },
    "usd_cad": {
        "known_id": "FXUSDCAD",
        "label_terms": ("usd/cad", "daily average"),
    },
    "corra": {
        "label_terms": ("canadian overnight repo rate average", "corra", "%"),
    },
    "t_bill_1m": {"label_terms": ("treasury bills", "1 month")},
    "t_bill_2m": {"label_terms": ("treasury bills", "2 month")},
    "t_bill_3m": {"label_terms": ("treasury bills", "3 month")},
    "t_bill_6m": {"label_terms": ("treasury bills", "6 month")},
    "t_bill_1y": {"label_terms": ("treasury bills", "1 year")},
    "goc_2y": {"label_terms": ("benchmark bond yield", "2 year")},
    "goc_5y": {"label_terms": ("benchmark bond yield", "5 year")},
    "goc_10y": {"label_terms": ("benchmark bond yield", "10 year")},
    "goc_30y": {"label_terms": ("benchmark bond yield", "long-term")},
}

BOS_EXCLUDED_SERIES = {
    "PC1",
    "BOS_ACTIVITY_INDICATOR",
    "BOS_PRICE_INDICATOR",
}
