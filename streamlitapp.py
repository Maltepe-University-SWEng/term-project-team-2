import streamlit as st
import torch
import torch.nn.functional as F
import os
import time
from main import CharTokenizer, SimpleLanguageModel

class FikraGenerator:
    def __init__(self, model_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.load_model(model_path, tokenizer_path)
    
    def load_model(self, model_path, tokenizer_path):
        # Tokenizer'ı yükle
        try:
            self.tokenizer = CharTokenizer.load(tokenizer_path)
            # Model'i yükle
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = SimpleLanguageModel(
                vocab_size=self.tokenizer.vocab_size,
                embedding_dim=64,
                hidden_dim=128
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            st.error(f"Model yüklenirken hata: {e}")
            return False
    
    def generate_fikra(self, prompt, max_length=100, temperature=0.5):
        """Verilen başlangıçla fıkra üret"""
        if not self.model or not self.tokenizer:
            return "Model yüklenemedi! Lütfen dosya yollarını kontrol edin."
        
        # Prompt'u uygun formata getir
        if not prompt.startswith("FIKRA:"):
            prompt = f"FIKRA: {prompt}"
        
        # Prompt'u tokenize et
        context = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        generated = list(context[0].cpu().numpy())
        
        # Metin üretimi
        with torch.no_grad():
            for _ in range(max_length):
                # Kontekst boyutunu sınırla
                if len(context[0]) > 150:
                    context = context[:, -150:]
                
                # Sonraki tokeni tahmin et
                outputs = self.model(context)
                next_token_logits = outputs[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Yeni tokeni ekle
                context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
                generated.append(next_token.item())
                
                # Bitiş kontrolleri
                if "\n\n" in self.tokenizer.decode(generated[-10:]):
                    break
                
                # Nokta, ünlem, soru işareti sonrası bitiş kontrol et
                if len(generated) > 100 and next_token.item() in [self.tokenizer.char_to_idx.get('.'), 
                                                              self.tokenizer.char_to_idx.get('!'),
                                                              self.tokenizer.char_to_idx.get('?')]:
                    if torch.rand(1).item() < 0.3:  # %30 olasılıkla bitir
                        break
        
        # Üretilen metni döndür
        return self.tokenizer.decode(generated)

# Streamlit arayüzü
def main():
    st.set_page_config(
        page_title="Türkçe Fıkra Üreteci",
        page_icon="😄",
        layout="centered"
    )
    
    st.title("🤣 Türkçe Fıkra Üreteci")
    st.markdown("""
    Bu uygulama, derin öğrenme kullanarak Türkçe fıkralar üretir. 
    Bir fıkra başlangıcı yazın, yapay zeka geri kalanını tamamlasın!
    """)
    
    # Model seçimi
    model_options = {
        "Prompt-Story Fıkra Modeli": {
            "model": "./model_data/finetune3_model/best_model.pt",
            "tokenizer": "./model_data/finetune3_model/tokenizer.pkl"
        },
        "Genel Fıkra Modeli": {
            "model": "./model_data/joke_model/best_model.pt",
            "tokenizer": "./model_data/joke_model/tokenizer.pkl"
        },
        "Temel Türkçe Modeli": {
            "model": "./model_data/base_model/model_epoch_9.pt",
            "tokenizer": "./model_data/base_model/tokenizer.pkl"
        }
    }
    
    selected_model = st.selectbox(
        "Kullanılacak modeli seçin:",
        list(model_options.keys())
    )
    
    # Seçilen modele göre dosya yollarını belirle
    model_path = model_options[selected_model]["model"]
    tokenizer_path = model_options[selected_model]["tokenizer"]
    
    # İlk çalıştırmada modeli yükle
    if "generator" not in st.session_state:
        with st.spinner("Model yükleniyor..."):
            st.session_state.generator = FikraGenerator(model_path, tokenizer_path)
    
    # Model değiştirildiğinde yeniden yükle
    if "last_model" not in st.session_state or st.session_state.last_model != selected_model:
        with st.spinner("Model değiştirildi, yeni model yükleniyor..."):
            st.session_state.generator = FikraGenerator(model_path, tokenizer_path)
            st.session_state.last_model = selected_model
    
    # Parametre kontrolleri
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider(
            "Yaratıcılık (Temperature):",
            min_value=0.1,
            max_value=1.5,
            value=0.5,
            step=0.1,
            help="Düşük değerler daha tutarlı, yüksek değerler daha yaratıcı çıktılar üretir"
        )
    
    with col2:
        max_length = st.slider(
            "Maksimum Uzunluk:",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Üretilecek metnin maksimum karakter sayısı"
        )
    
    # Hazır başlangıç önerileri
    prompt_suggestions = [
        "Temel bir gün",
        "Nasreddin Hoca",
        "İki arkadaş",
        "Adamın biri",
        "Öğretmen sınıfta"
    ]
    
    selected_suggestion = st.selectbox(
        "Hazır başlangıç seçin (veya aşağıya kendi başlangıcınızı yazın):",
        [""] + prompt_suggestions
    )
    
    # Metin girişi
    user_prompt = st.text_input(
        "Fıkra başlangıcınızı buraya yazın:",
        value=selected_suggestion
    )
    
    # Fıkra üretme butonu
    if st.button("Fıkra Üret", type="primary", disabled=not user_prompt):
        if not user_prompt:
            st.warning("Lütfen bir fıkra başlangıcı girin!")
        else:
            with st.spinner("Fıkra üretiliyor..."):
                # Daha uzun süren işlemlerde görsel geri bildirim için
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simüle edilmiş hesaplama zamanı
                    progress_bar.progress(i + 1)
                
                generated_fikra = st.session_state.generator.generate_fikra(
                    user_prompt, 
                    max_length=max_length,
                    temperature=temperature
                )
            
            # Sonucu göster
            st.success("Fıkra üretildi!")
            st.markdown("### Üretilen Fıkra")
            st.markdown(f"<div style='background-color: #1a237e; padding: 15px; border-radius: 5px;'>{generated_fikra}</div>", unsafe_allow_html=True)
            
            # Üretim parametrelerini göster
            st.info(f"Bu fıkra temperature={temperature}, max_length={max_length} ile üretilmiştir.")
    
    # Alt bilgi
    st.markdown("---")
    st.markdown("""
    **Not:** Bu uygulama, LSTM tabanlı dil modeli kullanarak fıkra üretir. Üretilen içerik tamamen yapay zeka tarafından oluşturulur.
    """)

if __name__ == "__main__":
    main()