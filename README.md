# Financial-Fraud
Financial Fraud Data
```{r}
library(dplyr)
library(ggplot2)
library(randomForest)
library(caret)
library(tidyverse)
library(psych)


set.seed(123)

# 1. Veriyi belirli bir jitter (gürültü) faktörü ile üretmek için fonksiyon tanımlıyoruz.
generate_dataset <- function(noise_factor) {
  n <- 2000  # Toplam 2000 gözlem (1000 hileli, 1000 hileli olmayan)
  
  # Hileli olmayan gözlemler
  non_fraud <- data.frame(
    DSRI = jitter(runif(n/2, 1.0, 1.3), factor = noise_factor),
    GMI  = jitter(runif(n/2, 0.9, 1.1), factor = noise_factor),
    AQI  = jitter(runif(n/2, 0.9, 1.1), factor = noise_factor),
    SGI  = jitter(runif(n/2, 1.0, 1.2), factor = noise_factor),
    DEPI = jitter(runif(n/2, 0.95, 1.05), factor = noise_factor),
    SGAI = jitter(runif(n/2, 0.95, 1.05), factor = noise_factor),
    LVGI = jitter(runif(n/2, 0.9, 1.1), factor = noise_factor),
    TATA = jitter(runif(n/2, -0.02, 0.02), factor = noise_factor),
    Class = 0
  )
  
  # Hileli gözlemler
  fraud <- data.frame(
    DSRI = jitter(runif(n/2, 1.3, 1.7), factor = noise_factor),
    GMI  = jitter(runif(n/2, 1.1, 1.4), factor = noise_factor),
    AQI  = jitter(runif(n/2, 1.1, 1.4), factor = noise_factor),
    SGI  = jitter(runif(n/2, 1.3, 1.8), factor = noise_factor),
    DEPI = jitter(runif(n/2, 1.3, 1.8), factor = noise_factor),
    SGAI = jitter(runif(n/2, 0.8, 0.95), factor = noise_factor),
    LVGI = jitter(runif(n/2, 1.1, 1.3), factor = noise_factor),
    TATA = jitter(runif(n/2, 0.03, 0.06), factor = noise_factor),
    Class = 1
  )
  
  # %5'lik rastgele etiket hatası ekleme:
  flipped_labels <- sample(1:n, size = 0.05 * n)
  # Hileli olmayanlardan hileli olan yap (ilk 1000 gözlem):
  non_fraud$Class[flipped_labels[flipped_labels <= n/2]] <- 1
  # Hilelilerden hileli olmayan yap (son 1000 gözlem):
  fraud$Class[flipped_labels[flipped_labels > n/2] - n/2] <- 0
  
  # Veriyi birleştirip karıştırıyoruz:
  data <- bind_rows(non_fraud, fraud) %>% slice(sample(1:n))
  
  return(data)
}

# 2. Grid search için gürültü (jitter) faktörleri ızgarasını tanımlıyoruz.
noise_grid <- seq(1, 3, by = 0.5)  # Örnek: 1, 1.5, 2, 2.5, 3

# Sonuçları saklamak için boş bir veri çerçevesi oluşturalım.
results <- data.frame(noise_factor = noise_grid, ROC = NA)

# 3. Model eğitimi için caret::train fonksiyonunu kullanacağız.
# Lojistik regresyon modeli (glm) ile 5 katlı çapraz doğrulama yapacağız.
# ROC (AUC) metriğini kullanarak performansı değerlendireceğiz.
control <- trainControl(method = "cv",
                        number = 5,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)

# Not: caret, sınıf değişkeninin factor olmasını bekler.
# Sınıf etiketlerini "Fraud" ve "NonFraud" olarak belirleyeceğiz.

for (i in seq_along(noise_grid)) {
  cat("Noise factor =", noise_grid[i], "\n")
  
  # Veriyi oluştur:
  data_i <- generate_dataset(noise_grid[i])
  
  # Sınıf değişkenini factor'a çevir:
  data_i$Class <- factor(ifelse(data_i$Class == 1, "Fraud", "NonFraud"))
  
  set.seed(123)  # Model yeniden üretilebilir olsun
  model <- train(Class ~ ., 
                 data = data_i, 
                 method = "glm", 
                 family = binomial,
                 trControl = control,
                 metric = "ROC")
  
  # En iyi ROC değerini kaydedelim:
  results$ROC[i] <- max(model$results$ROC)
}

print(results)

# En iyi performansa sahip gürültü faktörünü bulma:
best_noise <- results$noise_factor[which.max(results$ROC)]
cat("En iyi ROC değerini veren noise factor:", best_noise, "\n")

# Gürültü faktörü 1 ile veri oluşturma
best_noise_factor <- 1
noisy_data <- generate_dataset(best_noise_factor)

# 4. Grid search sonuçlarını görselleştirelim:
ggplot(results, aes(x = noise_factor, y = ROC)) +
  geom_line(color = "blue") +
  geom_point(color = "red", size = 3) +
  geom_vline(xintercept = best_noise, linetype = "dashed", color = "darkgreen") +
  labs(title = "Noise Factor vs. ROC (AUC)",
       x = "Noise Factor",
       y = "ROC (AUC)") +
  theme_minimal()

# Sınıf değişkenini faktör olarak ayarlama
noisy_data$Class <- factor(ifelse(noisy_data$Class == 1, "Fraud", "NonFraud"))

# Tanımlayıcı istatistikler (sayısal değişkenler için)
summary_stats <- describe(noisy_data[, -9])  # Class değişkenini hariç tutuyoruz

# Sonuçları görüntüleme
print(summary_stats)



# DSRI değişkeninin dağılımı
ggplot(noisy_data, aes(x = DSRI, fill = as.factor(Class))) +
  geom_density(alpha = 0.5) +
  ggtitle("DSRI Dağılımı (Gürültü Eklenmiş)")

# Eğitim ve test setlerine ayırma
set.seed(123)
trainIndex <- createDataPartition(noisy_data$Class, p = 0.7, list = FALSE)
trainData <- noisy_data[trainIndex, ]
testData <- noisy_data[-trainIndex, ]

# Gerekli paketleri yükleyin
library(randomForest)
library(caret)
library(pROC)

# Reprodüksiyon için sabit seed
set.seed(123)

# 1. Çapraz Doğrulama Ayarları
control <- trainControl(method = "cv", 
                        number = 5, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)

# 2. Genişletilmiş Hiperparametre Grid'i
tuneGrid <- expand.grid(.mtry = c(2, 3, 4, 5, 6))

# 3. Model Eğitimi (Gelişmiş Hiperparametre Optimizasyonu)
rfModel <- train(as.factor(Class) ~ ., 
                 data = trainData, 
                 method = "rf", 
                 metric = "ROC", 
                 trControl = control, 
                 tuneGrid = tuneGrid,
                 ntree = 500,            # Daha yüksek ağaç sayısı
                 importance = TRUE)
print(rfModel)
# 4. Test Seti Üzerinde Tahmin
predictions <- predict(rfModel, testData)

# 5. Performans Metrikleri
confMat <- confusionMatrix(predictions, as.factor(testData$Class))
print(confMat)

# ROC-AUC Hesaplama
roc_obj <- roc(as.numeric(testData$Class), as.numeric(predictions))
auc_value <- auc(roc_obj)
print(paste("AUC:", auc_value))

# Değişken Önem Düzeyleri
varImpPlot(rfModel$finalModel)




# Gerekli paketler
library(caret)
library(ggplot2)
library(reshape2)

# 1. Karışıklık Matrisini Hesaplama
confMat <- confusionMatrix(predictions, as.factor(testData$Class))
print(confMat)  # Metin tabanlı çıktı

# 2. Karışıklık Matrisini Görsel Hale Getirme
# Matris verisini uygun forma dönüştürme
cm_data <- as.table(confMat$table)
cm_df <- as.data.frame(cm_data)

# 3. ggplot2 ile Görselleştirme
ggplot(data = cm_df, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +  # Renk yoğunluğu
  geom_text(aes(label = Freq), vjust = 1, color = "black", size = 6) +  # Frekans etiketleri
  scale_fill_gradient(low = "lightblue", high = "darkblue") +  # Renk gradyanı
  labs(title = "Confusion Matrix",
       x = "Predicted",
       y = "Actual") +
  theme_minimal()


# Gerekli paketler
library(pROC)
library(ggplot2)

# Model tahmin olasılıkları (olasılık değerleriyle ROC eğrisi çizmek için)
# 'type = "prob"' ile sınıf olasılıklarını alıyoruz
pred_probs <- predict(rfModel, testData, type = "prob")

# ROC Eğrisi için olasılıkların pozitif sınıfa (örneğin 'Fraud') ait olanı kullanılır
roc_obj <- roc(response = testData$Class, predictor = pred_probs[, "Fraud"])

# ROC Eğrisi Çizimi
plot(roc_obj, 
     col = "#1c61b6", 
     lwd = 3, 
     main = "ROC Eğrisi (Random Forest Modeli)",
     print.auc = TRUE,               # AUC skorunu grafikte gösterir
     print.auc.cex = 1.2,            # AUC yazı boyutu
     print.auc.y = 0.4               # AUC pozisyonu
)

# İdeal Model İçin Referans Çizgisi (45° çizgi)
abline(a = 0, b = 1, lty = 2, col = "gray")  # Rastgele tahmin çizgisi


library(caret)

# Modelin karışıklık matrisini oluşturma
confMat <- confusionMatrix(predictions, as.factor(testData$Class))

# Performans metriklerini hesaplama
accuracy <- confMat$overall["Accuracy"]
precision <- confMat$byClass["Pos Pred Value"]
recall <- confMat$byClass["Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)
roc_auc <- confMat$byClass["Balanced Accuracy"]

# Sonuçları ekrana yazdırma
performance_metrics <- data.frame(
  Metric = c("Doğruluk (Accuracy)", "ROC-AUC Skoru", "F1 Skoru", "Hassasiyet (Precision)", "Duyarlılık (Recall)"),
  Değer = round(c(accuracy, roc_auc, f1_score, precision, recall), 3)
)

print(performance_metrics)


```

