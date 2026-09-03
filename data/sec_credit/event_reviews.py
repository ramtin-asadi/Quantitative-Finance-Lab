"""Auditable review overrides for SEC Item 1.03 filings.

The SEC item flag is retained unchanged in the output. These reviews distinguish
registrant bankruptcy petitions from subsidiary cases, notices, and later plan
updates after reading the primary filing document identified by the accession.
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DYNAMIC_REVIEWS = HERE / "cache" / "bankruptcy_reviews.parquet"

MANUAL_REVIEWS = {
    "0001193125-13-408263": (
        "plan_confirmation_or_emergence",
        False,
        "AMR/American plan confirmation; the Chapter 11 case began before the 2012 sample.",
    ),
    "0000732717-12-000006": (
        "metadata_miscoded",
        False,
        "AT&T credit-facility commitment reduction, not a bankruptcy filing.",
    ),
    "0001104659-20-077745": (
        "registrant_bankruptcy_petition",
        True,
        "Chesapeake Energy registrant and subsidiaries filed Chapter 11 petitions.",
    ),
    "0000895126-21-000016": (
        "plan_confirmation_or_emergence",
        False,
        "Chesapeake plan confirmation following the June 2020 petition.",
    ),
    "0000950157-19-000032": (
        "announced_intent_to_file",
        False,
        "PG&E announced an expected Chapter 11 filing; no petition had yet been filed.",
    ),
    "0001193125-19-019657": (
        "registrant_bankruptcy_petition",
        True,
        "PG&E Corporation and its utility subsidiary filed Chapter 11 petitions.",
    ),
    "0000950157-20-000795": (
        "plan_confirmation_or_emergence",
        False,
        "PG&E plan confirmation following the January 2019 petition.",
    ),
    "0001104659-17-039161": (
        "subsidiary_bankruptcy_petition",
        False,
        "NRG subsidiary GenOn and affiliates, not NRG Energy, filed Chapter 11.",
    ),
    "0001104659-17-073701": (
        "plan_confirmation_or_emergence",
        False,
        "GenOn subsidiary plan confirmation following its June 2017 petition.",
    ),
    "0001031296-18-000024": (
        "subsidiary_bankruptcy_petition",
        False,
        "FirstEnergy Solutions and related subsidiaries, not FirstEnergy, filed Chapter 11.",
    ),
    "0001415404-26-000038": (
        "subsidiary_bankruptcy_petition",
        False,
        "EchoStar subsidiary Hughes Satellite Systems and affiliates filed Chapter 11.",
    ),
}


def load_reviews() -> dict[str, tuple[str, bool, str]]:
    reviews: dict[str, tuple[str, bool, str]] = {}
    if DYNAMIC_REVIEWS.exists():
        frame = pd.read_parquet(
            DYNAMIC_REVIEWS,
            columns=[
                "accession",
                "bankruptcy_scope",
                "is_registrant_bankruptcy_event",
                "bankruptcy_review_note",
            ],
        )
        reviews.update(
            {
                str(row.accession): (
                    str(row.bankruptcy_scope),
                    bool(row.is_registrant_bankruptcy_event),
                    str(row.bankruptcy_review_note),
                )
                for row in frame.itertuples(index=False)
            }
        )
    reviews.update(MANUAL_REVIEWS)
    return reviews


BANKRUPTCY_REVIEWS = load_reviews()
