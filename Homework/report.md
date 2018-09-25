# Human emotion recognition

## Problem definition
本次練習所使用之資料集和 `Kaggle Facial Expression Recognition Challenge (FER2013)` 是類似的，
但由於本次測驗的基本要求，無法使用這裡的數據集進行擴充的動作。題目定義簡潔:
給予一張圖片，將其歸類至7種表情中的其中一種。可以將其視為一個 multi-class classification 的問題。
原題目希望透過表情，可以大略地知道該受測對象的情緒狀態，並可對其做出進一步的回應。<br>

![image info](front_page.PNG) <br>
*Kaggle Facial Expression Recognition Challenge*

例如，商業，客戶在接觸到特定產品後的表情變化，可做為產品未來改進方向、目標客群選擇的參考依據。
與文字、語音的用戶回饋相比，這是更容易取得的資料來源。卻也是較難分析的一種數據型式，
畢竟，就算是人類也常有誤判他人情緒的狀況。
而這樣的題目，也是期望在基於表情的情緒分析上，探究機器是否也有能超越人類辨識率的可能性。

## Provided dataset
在資料集的部分，總共有 22000 筆 training data，每筆資料為 48*48、灰階化處理、人臉置中過後的圖片。
若將圖片視覺化呈現出來，可以發現資料前處理的工作已被大量簡略，但圖片的品質並非都十分理想。
像是臉部的浮水印、圖片過度曝光/亮度不足導致臉部輪廓不明顯、甚至是缺圖(sample_59)等狀況。

![image info](sample_115.PNG) <br>
*image with watermark*

![image info](sample_59.PNG) <br>
*image unrelated*


所幸這類資料佔全體資料的比例並不高(個人估計不超過3%)，在實際運用上並不會對 model training 造成太大的衝擊。

另一個較特別的部分，這是個 imbalance dataset，各個 class 的比例並不平均。
`"disgust"` 的比例佔全體約 1%，幾乎是可以被忽略的量。

![image info](classes_ratio.PNG) <br>
*the ratio of each class in training set*

在實務上，若該 class 並不是特別重要，或是這個 task 預測錯誤的 penalty 不大
(如先前課程中 precision, recall 所講的概念)。
那麼，其實直接將該 class 的資料無視，或是併入其他 class 中，會是個不錯的選擇。
如此一來便能夠避免 model 為了 `class1: "disgust"` 的預測率，而拖累了其他 class 預測的精度。

## Data preprocessing
由於此次練習所用之資料即已做了不少前處理，因此我並沒有特別做甚麼動作。
一般而言，在影像處理的task 都必須做:
* 將目標 image 從原始圖片中裁切出來，或是將其置中於原圖片
* 將圖片做 normalization，以本次的資料集為例，所有值都介於 0-255 之間(有助於 training)
* 將圖片裁剪/縮放成相同大小(部分特化的 model 可以省略此步驟 e.g. global pooling)
* 去除資料中的雜訊。例如，去除與此次 task 不相干的圖片、圖片中有部分區塊被遮蔽等問題

### Data augmentation
有鑑於 model performance 難以提升，一個簡單的做法就是增加資料量
`keras.preprocessing.image.ImageDataGenerator` 有大部分常見的資料擴增方式，
而我使用到的部分則是:
* brightness_range
* shear_range
* zoom_range
* rotation_range
* horizontal_flip

![image info](image_aug1.PNG) <br>
*data augmentation with only one feature*  <br><br>

![image info](image_aug2.PNG) <br>
*data augmentation with multiple features*

一個簡單的準則: `"more training data" is preferable than "more complex model"`<br>
但 model 也不能過度簡單，以至於無法完整表達原始資料(實驗中調整)


## Model building
原則上參考VGG 的架構進行網路建置: <br>
`filter=3*3, CONV -> POOL ... Flatten -> FC-> Prediction`

第一次建置的模型如下:
8 8 16 16 32 32 128 7
accuracy 約0.55 但是有點 overfitting (train/valid acc 相差約5%)

探究其原因，發現相較於資料量，可調整的參數量有點過多

根據不具名的都市傳說 `建議參數量 <= 10 * training_data_size`

嘗試 8 8 16 16 32 32 64 7
accuracy 約0.58 且 overfitting 有些許改善

然而，這樣的結果並不理想，甚至可視為 underfitting
於是 data augmentation 便派上用場了，將training set 擴增5倍

嘗試 8 8 16 16 32 32 64 7
accuracy 沒有明顯改善，依舊underfitting -> go deeper

嘗試 16 16 32 32 64 64 64 7   並且將 epoch 提升到300~500
accuracy 拉升到 0.7 左右，並以此作為最終 model

### Model validation
雖然 keras 所提供的 model.fit() 有提供 validation_split。
但為了徹底地避免 data leakage，我仍自行將資料切分為 training set & validation set。
在決定最終的 model 前，絕不使用後者進行驗證

從 predict_prob 可以發現，部分難以區分的表情，看似多種不同情緒的組合。
實際上也是如此，人類的情緒也很難用單純的 0/1 作區格。
此次 task 可能是為了簡化問題，才將其設定為 0/1 的 accuracy metric

在confusion matrix 中也看得出來<br>
[[305   1   49  49  103 10  95]<br>
 [13    31  3   5   13   1   5]<br>
 [ 63   2 237  32 162  66  76]<br>
 [ 30   0  18 941  32  23  62]<br>
 [ 83   0  67  35 405  10 138]<br>
 [ 25   0  40  29   9 351  21]<br>
 [ 41   0  17  70 109  17 506]]<br>

 部分 class prediciton 出現混雜的狀況，而我們也可以針對這些資訊，對 model 做進一步的優化。

## Room for improvement
* 更多不同型式的 data augmentation
* 嘗試更進階的 CNN 架構
* 用特殊工具座前處理(例如，將浮水印去除)
* 針對較難區分的 case 特別建立 model
* 用其他 task 的 pretrain model (此task 並不允許)
* 用其他外部資料建立新的training set (此task 並不允許)
