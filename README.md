BINANCE FUTURE ML TRADER
------------------------------------------------------------------------------------------------------------------
Files to create
config.ini

PostgreSQL installation required.

apt install git
apt install docker
apt install docker-compose
install anaconda

You need to create Anaconda env (Python 3.11.7) and install requirements.txt into it.

When adding a new coin to the coin list, precisionda must be added. Specifies the decimal number of precision quantity.
When adding coins, attention must be paid to the minimum quantity! (scripts/quantity_calculator can be run and the results can be viewed in the db.)

DB schemas should be created with scripts/create_db_schema.

New coin addition should be made to top_n_coine. After this is updated, the selected coin can be added to the selected coin according to the minimum quantity in the quantities table by running quantity_calculator.

Note: When pulling data from the technical side of the DB, the data may not appear in a direct order, so it should be sorted according to the index.


https://cloud.google.com/sdk/docs/install

gcloud auth application-default login

----------------------------------------------------------------------------------------------------------------------
Oluşturulacak dosyalar
config.ini

PostgreSQL kurulumu gerekiyor.

apt install git
apt install docker
apt install docker-compose
install anaconda

Anaconda env(Python 3.11.7) oluşturup içine requirements.txt nin kurulması gerekiyor.

Coin listesine yeni coin eklerken precisionda eklenmesi gerekiyor. Precision quantity nin ondalık sayısını belirtiyor.
Coin eklenirken minimum quantity miktarına dikkat edilmesi gerekiyor!(scripts/quantity_calculator çalışıtırılıp sonuçlara dbden bakılabilir.)

scripts/create_db_schema ile db şemaları oluşturulmalı.

Yeni coin eklemesi top_n_coine yapılmalıdır. Burası güncellendikten sonra, quantity_calculator çalıştırılarak quantities tablosundaki minimum quantitye göre selected coine ekleme yapılabilir.

Not: DB'de technical taraflarından veri çekerken veri sıralı gelmeyebilir direkt, ona göre indexe göre sıralama yapılarak çekilmeli.


https://cloud.google.com/sdk/docs/install

gcloud auth application-default login

