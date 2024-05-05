Oluşturulacak klasörler 
/app/ml_models/coin

Oluşturulacak dosyalar
config.ini

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

log/simulation.log
log/general.log eklenecek.
config.ini içerisindeki db bağlantılarında localhost değişebilir.

Deployment Steps:(Eski)

docker build -t btrder_image .

docker network create my-network

docker run --name btrder --network my-network -d btrder_image:latest

docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=qwertypoikjh1 -e POSTGRES_DB=btc --network my-network -d postgres:12.17-alpine3.19

docker run --name postgres -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=qwertypoikjh1 -e POSTGRES_DB=btc --network my-network -p 5433:5432 -d postgres:12.17-alpine3.19



https://cloud.google.com/sdk/docs/install

gcloud auth application-default login

