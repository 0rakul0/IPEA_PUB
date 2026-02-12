from ingestao import create_ingestion
from db.banco_metadados import MetadataDB
from scraper import Scraper

import time
import os

TOTAL_PAGES = 860

db_metadata = MetadataDB()

scraper = Scraper()
for i in range(1, 860):
    scraper.processar_pagina(i)

time.sleep(5)

removidos = db_metadata.remover_duplicatas()
print(f"{removidos} registros duplicados removidos.")

if os.name == "nt":
    os.system('cls')


while True:
    sucesso = create_ingestion.processar_documento()
    if not sucesso:
        break

print("Pipeline concluído.")



