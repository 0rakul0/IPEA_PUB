from db.banco_metadados import MetadataDB
from scraper import Scraper

TOTAL_PAGES = 860

db_metadata = MetadataDB()

scraper = Scraper()
for i in range(1, TOTAL_PAGES):
    scraper.processar_pagina(i)

removidos = db_metadata.remover_duplicatas()
print(f"{removidos} registros duplicados removidos.")
