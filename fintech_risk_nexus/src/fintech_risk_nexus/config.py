from typing import Dict, List

COUNTRY_CODE_TO_NAME: Dict[str, str] = {
    "NG": "Nigeria",
    "KE": "Kenya",
    "GH": "Ghana",
    "ZA": "South Africa",
    "UG": "Uganda",
    "TZ": "Tanzania",
    "RW": "Rwanda",
    "ET": "Ethiopia",
    "CM": "Cameroon",
    "ZM": "Zambia",
}

DEFAULT_COUNTRIES: List[str] = list(COUNTRY_CODE_TO_NAME.keys())

# pytrends "geo" expects ISO 3166-1 alpha-2 codes for country-level queries
GEO_MAP: Dict[str, str] = {code: code for code in DEFAULT_COUNTRIES}

# Representative fintech/mobile money brands per country
COUNTRY_BRANDS: Dict[str, List[str]] = {
    "NG": ["Flutterwave", "Paystack", "OPay", "Paga", "Interswitch"],
    "KE": ["M-Pesa", "Airtel Money", "Equitel", "KCB M-Pesa", "T-Kash"],
    "GH": ["MTN MoMo", "Vodafone Cash", "AirtelTigo Money", "ExpressPay"],
    "ZA": ["SnapScan", "Ozow", "TymeBank", "Capitec App", "FNB eWallet"],
    "UG": ["MTN MoMo", "Airtel Money", "MoMoPay", "Centenary Mobile"],
    "TZ": ["M-Pesa", "Tigo Pesa", "Airtel Money", "HaloPesa"],
    "RW": ["MTN MoMo", "Airtel Money", "BK App"],
    "ET": ["Telebirr", "CBE Birr", "HelloCash"],
    "CM": ["Orange Money", "MTN MoMo", "Express Union Mobile"],
    "ZM": ["MTN MoMo", "Airtel Money", "Zamtel Kwacha"],
}
