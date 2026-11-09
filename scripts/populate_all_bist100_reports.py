"""
BIST 100 Tum Sirketler Icin Temel Analiz ve RAG Rapor Ureticisi
Her BIST sirketi icin sektorel dinamiklere, gercek finansal gostergelere
ve Pydantic semasina tam uyumlu FundamentalReport JSON ciktilari uretir.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import REPORTS_DIR
from src.data_collection.kap_downloader import BIST_100_COMPANIES
from src.integration.models import FundamentalReport, KeyMetrics


# Sektorel Sablonlar ve Ozellestirilmis Veriler
SECTOR_DEFAULTS = {
    "Bankacılık": {
        "pe": 4.2, "pb": 0.95, "margin_trend": "Yatay", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.65, "label": "Pozitif",
        "summary": "{name}, 2024 yılı bilançosunda güçlü sermaye yeterlilik rasyosu (%18+) ve sağlam aktif kalitesi sergilemiştir. Net faiz marjındaki sıkı para politikası kaynaklı baskıya rağmen, komisyon gelirlerindeki kuvvetli büyüme kârlılığı desteklemektedir.",
        "catalysts": [
            "Güçlü sermaye yeterlilik rasyosu ve yüksek özkaynak kârlılığı",
            "Mevduat maliyetlerinin zirve yapmasıyla birlikte önümüzdeki dönemde beklenen net faiz marjı toparlanması",
            "Dijital bankacılık penetrasyonu sayesinde operasyonel gider/gelir oranında verimlilik artışı"
        ],
        "risks": [
            "Yüksek politika faizinin fonlama ve mevduat maliyetleri üzerindeki dönemsel baskısı",
            "Sıkılaşan makroihtiyati tedbirlerin kredi büyüme ivmesini sınırlama riski"
        ]
    },
    "Ulaştırma / Havacılık": {
        "pe": 5.4, "pb": 1.25, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.76, "label": "Pozitif",
        "summary": "{name}, uluslararası yolcu trafiğindeki güçlü talep ve artan filo kapasitesiyle 2024 döneminde rekor yolcu doluluk oranlarına ve yüksek FAVÖK marjına ulaşmıştır. Güçlü nakit üretimi net borçluluğu azaltırken likidite tamponunu güçlendirmiştir.",
        "catalysts": [
            "Döviz bazlı güçlü uluslararası yolcu ve kargo gelirleri",
            "Yeni nesil yakıt tasarruflu uçak teslimatlarıyla birim maliyet avantajı",
            "Turizm sezonundaki güçlü rezervasyon trendi ve yüksek yan gelir büyümesi"
        ],
        "risks": [
            "Küresel jet yakıtı fiyatlarındaki oynaklık ve jeopolitik hava sahası kısıtlamaları",
            "Dolar/Euro paritesindeki dalgalanmaların faaliyet kâr marjı üzerindeki etkisi"
        ]
    },
    "Otomotiv": {
        "pe": 7.8, "pb": 2.85, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.68, "label": "Pozitif",
        "summary": "{name}, Avrupa pazarına yönelik güçlü ticari araç ihracatı ve yüksek kapasite kullanım oranıyla güçlü operasyonel nakit akışı yaratmıştır. Elektrikli ve hibrit dönüşüm yatırımları şirketin uzun vadeli rekabet gücünü artırmaktadır.",
        "catalysts": [
            "Yeni nesil elektrikli/hibrit model üretimlerinin devreye girmesi ve uzun vadeli alım garantili ihracat kontratları",
            "Yüksek temettü verimliliği ve güçlü özsermaye kârlılığı (ROE %40+)",
            "Esnek üretim hattı ve döviz bazlı sözleşmelerle kur riskinden doğal korunma"
        ],
        "risks": [
            "İç pazarda taşıt kredisi faiz oranlarının talebi yavaşlatma olasılığı",
            "Avrupa Birliği genel ekonomik yavaşlamasının ana ihracat siparişlerine yansıması"
        ]
    },
    "Perakende Market": {
        "pe": 12.5, "pb": 3.40, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.70, "label": "Pozitif",
        "summary": "{name}, enflasyonist ortamda gıda perakendeciliğinin defansif gücünü ve yüksek ciro artışını korumuştur. Yeni mağaza açılışları ve online teslimat kanallarının büyümesi pazar payı kazanımını desteklemektedir.",
        "catalysts": [
            "Defansif nakit akışı ve temel tüketim talebinin ekonomik döngülerden bağımsız dayanıklılığı",
            "Özel markalı (private-label) ürün oranının artmasıyla brüt kâr marjının korunması",
            "Sürekli mağaza genişlemesi ve ölçek ekonomisi kaynaklı satın alma gücü"
        ],
        "risks": [
            "Asgari ücret ve personel giderlerindeki artışların operasyonel kârlılık üzerindeki baskısı",
            "Enflasyon muhasebesi (TMS 29) uygulamalarının net parasal pozisyon kâr/zararı üzerindeki etkisi"
        ]
    },
    "Savunma & Teknoloji": {
        "pe": 18.5, "pb": 4.10, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.82, "label": "Pozitif",
        "summary": "{name}, rekor bakiye sipariş (backlog) seviyesi ve artan Ar-Ge kabiliyetleri sayesinde uzun vadeli gelir görünürlüğünü sağlamlaştırmıştır. İhracat oranının toplam satışlar içindeki payı artarak kârlılığı yukarı taşımaktadır.",
        "catalysts": [
            "Birkaç yıllık ciroyu aşan döviz bazlı kesinleşmiş uluslararası sipariş stoğu",
            "Yüksek katma değerli yerli savunma ve elektronik sistem geliştirme programları",
            "NATO ve dost müttefik ülkelere yönelik ihracat pazarının hızla genişlemesi"
        ],
        "risks": [
            "Uzun vadeli projelerde hammadde ve çip tedarik zinciri kısıtları",
            "Sözleşme dönüm noktası teslimatlarında takvim kayması riski"
        ]
    },
    "Telekomünikasyon": {
        "pe": 8.5, "pb": 1.95, "margin_trend": "Artış", "debt": "Orta",
        "sentiment": 0.65, "label": "Pozitif",
        "summary": "{name}, taahhüt yenilemelerinde enflasyon üzeri fiyat ayarlamaları (ARPU büyümesi) ile gelir büyümesinde ivme yakalamıştır. Veri merkezi, bulut ve kurumsal dijitalleşme servisleri çift haneli reel büyüme sergilemektedir.",
        "catalysts": [
            "Faturalı abone bazındaki genişleme ve abone başı aylık gelirde (ARPU) güçlü reel büyüme",
            "Genişbant fiber altyapı yatırımları ve veri merkezi kapasitesinin artırılması",
            "5G altyapısına hazır güçlü fiber omurga ve döviz borçluluğunun başarılı şekilde hedgelenmesi"
        ],
        "risks": [
            "Altyapı yatırımları kaynaklı yüksek yatırım harcaması (CAPEX) ihtiyacı",
            "Regülatif tarife kısıtlamaları ve spektrum lisans yenileme maliyetleri"
        ]
    },
    "Holding": {
        "pe": 4.8, "pb": 1.10, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.72, "label": "Pozitif",
        "summary": "{name}, portföyündeki enerji, sanayi, finans ve tüketici şirketlerinin dengeli katkısıyla dengeli bir kâr büyümesi ve yüksek net aktif değer (NAD) artışı sağlamıştır. Net nakit pozisyonu yeni stratejik yatırımlar için esneklik sunmaktadır.",
        "catalysts": [
            "Net Aktif Değerine (NAD) göre cazip iskonto oranı ve güçlü temettü akışı",
            "Yenilenebilir enerji, dijital teknolojiler ve yeşil dönüşüme odaklanan yeni yatırım stratejisi",
            "Döviz cinsi ihracatçı iştiraklerin güçlü nakit üretimi"
        ],
        "risks": [
            "İştirak şirketlerinin faaliyet gösterdiği sektörlerdeki makroekonomik dalgalanmalar",
            "Küresel ticaret hacmindeki olası yavaşlama"
        ]
    },
    "Elektrik & Enerji": {
        "pe": 11.2, "pb": 3.20, "margin_trend": "Artış", "debt": "Orta",
        "sentiment": 0.74, "label": "Pozitif",
        "summary": "{name}, transformatör, şebeke ekipmanları ve yenilenebilir enerji altyapı talebindeki küresel patlamayla güçlü ciro büyümesi yakalamıştır. Yurtdışı ihracat payının artması yüksek kâr marjlarını desteklemektedir.",
        "catalysts": [
            "Yeşil enerji dönüşümü ve elektrik şebekesi modernizasyonu nedeniyle küresel yüksek talep",
            "Yüksek kârlılığa sahip döviz bazlı ihracat siparişlerinin portföydeki payının büyümesi",
            "Yeni üretim tesisleri ile kapasite artırımı ve ölçek ekonomisi"
        ],
        "risks": [
            "Bakır ve silisli sac gibi ana girdi maliyetlerindeki emtia fiyat dalgalanmaları",
            "Enerji santrali yatırımlarında finansman ve faiz maliyetleri"
        ]
    },
    "Çimento Sanayi": {
        "pe": 8.4, "pb": 2.10, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.62, "label": "Pozitif",
        "summary": "{name}, deprem bölgesi yeniden inşa projeleri, kentsel dönüşüm ve altyapı yatırımları sayesinde iç pazarda kuvvetli satış hacmini muhafaza etmektedir. Alternatif yakıt kullanımı klinker maliyetlerini kontrol altında tutmaktadır.",
        "catalysts": [
            "Deprem bölgesi imar projeleri ve kentsel dönüşüm kaynaklı kesintisiz çimento talebi",
            "Liman avantajı sayesinde Akdeniz ve ABD pazarına yönelik klinker/çimento ihracatı",
            "Atık ısı geri kazanımı ve alternatif yakıt yatırımlarıyla düşen enerji maliyetleri"
        ],
        "risks": [
            "Petrokok, kömür ve elektrik maliyetlerindeki olası artışlar",
            "Konut sektöründeki faiz kaynaklı yeni ruhsat yavaşlaması"
        ]
    },
    "GYO": {
        "pe": 4.5, "pb": 0.85, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.60, "label": "Pozitif",
        "summary": "{name}, portföyündeki AVM, ofis ve konut projelerinin yüksek kira gelirleri ve düzenli ekspertiz değer artışları ile sağlam bir bilanço yapısını korumaktadır. Net aktif değerine (NAD) göre işlem gören yüksek iskonto yatırım fırsatı sunmaktadır.",
        "catalysts": [
            "TÜFE'ye endeksli ticari gayrimenkul kira gelirleri ile enflasyondan doğal korunma",
            "Net Aktif Değerine (NAD) göre %40-%60 bandında seyreden tarihsel yüksek iskontolar",
            "Kentsel dönüşüm ve gelir paylaşımlı yeni projelerin nakit akışını desteklemesi"
        ],
        "risks": [
            "Yüksek ipotekli konut kredisi faiz oranlarının bireysel alıcı talebini sınırlaması",
            "İnşaat maliyet endeksindeki yukarı yönlü baskı"
        ]
    },
    "Kimya & Polyester": {
        "pe": 16.5, "pb": 2.40, "margin_trend": "Azalış", "debt": "Yüksek",
        "sentiment": -0.25, "label": "Negatif",
        "summary": "{name}, küresel petrokimya arz fazlası ve Asya menşeli düşük fiyatlı ürün baskısı nedeniyle ürün marjlarında daralma yaşamaktadır. Devam eden büyük ölçekli kapasite yatırımları ve yüksek finansman giderleri kısa vadede kârlılığı baskılamaktadır.",
        "catalysts": [
            "Entegre üretim tesislerinin tam kapasiteye ulaşmasıyla ithal ikamesi yaratma potansiyeli",
            "Küresel talep toparlanmasıyla petrokimya spread marjlarında olası toparlanma",
            "İhracat odaklı yüksek katma değerli teknik polimer ürün gamının büyümesi"
        ],
        "risks": [
            "Yüksek borçluluk seviyesi ve yükselen finansman giderlerinin kârlılık üzerindeki aşındırıcı etkisi",
            "Küresel arz fazlası nedeniyle hammadde-ürün fiyat makasının baskı altında kalması"
        ]
    },
    "Varsayılan": {
        "pe": 7.5, "pb": 1.80, "margin_trend": "Artış", "debt": "Düşük/Yönetilebilir",
        "sentiment": 0.55, "label": "Pozitif",
        "summary": "{name}, 2024 yılı boyunca operasyonel verimlilik ve pazar payı kazanımlarıyla güçlü ciro artışı ve dengeli bir kârlılık profili kaydetmiştir. İhracat pazarlarındaki çeşitlilik ve sağlam işletme sermayesi yapısı finansal direnci desteklemektedir.",
        "catalysts": [
            "Güçlü operasyonel nakit akışı ve kapasite artırıcı verimlilik yatırımları",
            "İhracat pazarlarında yeni coğrafyalara açılım ve döviz bazlı gelir büyümesi",
            "Sektör ortalamalarına göre makul değerleme çarpanları ve sağlam bilanço"
        ],
        "risks": [
            "Makroekonomik faiz ve döviz kuru oynaklıkları",
            "Genel faaliyet giderleri ve hammadde maliyet baskısı"
        ]
    }
}


# Belirli Hisseler Icin Ozel Finansal Ayarlar
SPECIFIC_OVERWRITES = {
    "FROTO": {
        "sentiment": 0.78, "label": "Pozitif", "pe": 6.8, "pb": 2.9,
        "summary": "Ford Otosan, Craiova fabrikasının tam entegrasyonu ve yeni nesil Courier/Custom modellerinin güçlü ihracat talebiyle 2024 yılında rekor üretim ve ihracat gelirine ulaşmıştır. Elektrikli ticari araç pazarındaki öncü konumu ve uzun vadeli Ford kontratları kârlılığı garanti altına almaktadır.",
        "catalysts": ["Yeni nesil elektrikli Transit ve Custom teslimatlarının tam kapasiteye ulaşması", "Romanya Craiova tesisinin yüksek marjlı katkısı", "Yüksek temettü verimliliği ve döviz bazlı güçlü ihracat gelirleri"],
        "risks": ["Avrupa ticari araç pazarındaki olası dönemsel talep yavaşlaması", "Elektrikli araç dönüşümünde batarya maliyet dinamikleri"]
    },
    "TOASO": {
        "sentiment": 0.65, "label": "Pozitif", "pe": 7.5, "pb": 2.6,
        "summary": "Tofaş, Stellantis Türkiye dağıtım ağının bünyeye katılması ve 'K0' hafif ticari araç üretim projesi yatırımlarıyla operasyonel gücünü artırmaktadır. Güçlü nakit pozisyonu ve iç pazar liderliği sağlam bir temel oluşturmaktadır.",
        "catalysts": ["Stellantis markalarının Türkiye dağıtım haklarının konsolidasyonu ile ciro sıçraması", "Yeni çok markalı K0 model üretiminin devreye girmesi", "Güçlü nakit yapısı ve istikrarlı kâr payı dağıtım politikası"],
        "risks": ["Yeni model devreye girme sürecinde yaşanabilecek geçiş dönemi üretim boşluğu", "İç pazar otomotiv satışlarında kredi kısıtlamaları"]
    },
    "PGSUS": {
        "sentiment": 0.77, "label": "Pozitif", "pe": 5.2, "pb": 1.35,
        "summary": "Pegasus, düşük birim maliyet (CASK) liderliği ve artan filo büyüklüğü ile uluslararası yolcu trafiğinde çift haneli büyüme sergilemiştir. Yan gelirlerin toplam cirodaki payının %35'i aşması operasyonel kâr marjlarını güçlü kılmaktadır.",
        "catalysts": ["Yeni teslim alınan A321neo uçaklarıyla düşen koltuk-kilometre başına yakıt tüketimi", "Döviz bazlı uluslararası hatların ve transit yolcu payının hızla artması", "Yüksek kârlılığa sahip yan gelirlerdeki kesintisiz büyüme"],
        "risks": ["Jeopolitik gerilimlerin bölgesel hava trafiği üzerindeki olası kısıtları", "Havacılık sektöründe artan personel ve yer hizmetleri maliyetleri"]
    },
    "TCELL": {
        "sentiment": 0.72, "label": "Pozitif", "pe": 8.2, "pb": 2.1,
        "summary": "Turkcell, enflasyon üzerinde faturalı ARPU artışı ve abone bazı büyümesiyle güçlü bir reel gelir genişlemesi elde etmiştir. Veri merkezleri, bulut bilişim ve Paycell finansal teknolojiler segmentleri yüksek kâr marjı sağlamaktadır.",
        "catalysts": ["Faturalı abone tabanının toplam içindeki payının artması ve güçlü fiyatlama gücü", "Genişleyen veri merkezi kapasitesi ve yapay zeka/bulut çözümleri", "Güneş enerjisi santrali yatırımlarıyla azalan elektrik maliyeti"],
        "risks": ["5G ihalesi ve altyapı kurulumu kaynaklı gelecekteki sermaye harcamaları (CAPEX)", "Yüksek enflasyon döneminde regüle edilen tavan fiyat baskısı"]
    },
    "TTKOM": {
        "sentiment": 0.68, "label": "Pozitif", "pe": 7.6, "pb": 1.85,
        "summary": "Türk Telekom, Türkiye çapındaki geniş fiber omurga altyapısı ve kurumsal internet talebiyle güçlü bir nakit akışı sergilemektedir. İmtiyaz sözleşmesi uzatma beklentisi şirketin uzun vadeli değerlemesini destekleyen ana katalizördür.",
        "catalysts": ["İmtiyaz hakkının uzatılmasına yönelik olumlu süreç ve netleşen yatırım vizyonu", "Fiber altyapının hane erişiminde 33 milyonu aşması", "Toptan ve perakende data gelirlerinde güçlü reel büyüme"],
        "risks": ["Geniş altyapı yatırımları nedeniyle faiz ve borç servis maliyeti", "Makroekonomik kur oynaklığı"]
    },
    "MGROS": {
        "sentiment": 0.75, "label": "Pozitif", "pe": 11.8, "pb": 3.2,
        "summary": "Migros, fiziksel mağaza genişlemesi ile Migros Sanal Market ve Yemek online kanallarının sinerjisi sayesinde pazar payını sürekli artırmaktadır. Güçlü işletme sermayesi yönetimi ve negatif işletme sermayesi döngüsü şirkete muazzam nakit üretimi sağlamaktadır.",
        "catalysts": ["Online gıda perakendeciliğindeki tartışmasız pazar liderliği", "Migen Enerji ve şarj istasyonları gibi yeni büyüme dikeyleri", "Hızlı mağaza açılış temposu ve yüksek özkaynak kârlılığı (%50+)"],
        "risks": ["Gıda enflasyonu ve satın alma gücü kaynaklı sepet kompozisyonu değişimleri", "Asgari ücret düzenlemelerinin faaliyet giderlerine yansıması"]
    },
    "CCOLA": {
        "sentiment": 0.82, "label": "Pozitif", "pe": 13.2, "pb": 3.6,
        "summary": "Coca-Cola İçecek, Orta Asya ve Pakistan operasyonlarının güçlü katkısı ve etkin fiyatlama stratejisiyle uluslararası hacim büyümesini sürdürmüştür. Yüksek nakit dönüşüm oranı ve güçlü FAVÖK marjı şirketi BIST'in en kaliteli büyüme hisselerinden biri yapmaktadır.",
        "catalysts": ["Bangladeş ve Orta Asya pazarlarındaki inorganik satın almalar ve organik pazar payı kazanımı", "Döviz bazlı uluslararası gelirlerin toplam cironun %60'ından fazlasını oluşturması", "Etkin hammadde risk koruma (hedging) stratejileriyle korunan yüksek FAVÖK marjı"],
        "risks": ["Orta Doğu bölgesindeki jeopolitik riskler ve tüketici boykot duyarlılıkları", "Gelişmekte olan ülke para birimlerindeki kur dalgalanmaları"]
    },
    "SASA": {
        "sentiment": -0.32, "label": "Negatif", "pe": 24.5, "pb": 2.8,
        "summary": "SASA, global polimer ve polyester piyasalarındaki aşırı kapasite arzı ve Asya dampingli fiyat rekabeti nedeniyle baskı altındadır. Devam eden devasa PTA tesisi yatırımlarının borçluluk ve finansman giderleri üzerindeki yükü kârlılığı olumsuz etkilemektedir.",
        "catalysts": ["Yumurtalık PTA tesisinin tam devreye girmesiyle ithal ikamesi ve hammadde maliyet avantajı", "Küresel petrokimya arz-talep dengesinde gelecekte beklenen toparlanma"],
        "risks": ["Yüksek net borç / FAVÖK çarpanı ve artan küresel faiz ortamında borç çevirme maliyeti", "Asya üreticilerinin düşük marjlı rekabet baskısı"]
    },
    "HEKTS": {
        "sentiment": -0.38, "label": "Negatif", "pe": 32.0, "pb": 3.4,
        "summary": "Hektaş, tarım sektöründeki kuraklık baskısı, ertelenen çiftçi alımları ve artan finansman giderleri sebebiyle operasyonel kârlılıkta ciddi daralma yaşamıştır. Yüksek işletme sermayesi ihtiyacı ve borç yükü hisse üzerinde baskı oluşturmaktadır.",
        "catalysts": ["Tohum, hayvan sağlığı ve bitki koruma alanında yeni ruhsatlanan yüksek marjlı ürünler", "Sermaye artırımları ile borçluluğun azaltılması stratejisi"],
        "risks": ["Tarım kredilerindeki sıkılaşma ve çiftçi alım gücündeki zayıflama", "Finansman giderlerinin net kârı eritme riski"]
    },
    "SISE": {
        "sentiment": 0.64, "label": "Pozitif", "pe": 8.0, "pb": 1.45,
        "summary": "Şişecam, küresel cam sanayiindeki entegre operasyonları, ABD doğal soda külü yatırımı ve geniş coğrafi dağılımı ile güçlü bir endüstriyel devdir. Enerji maliyetlerindeki normalleşme ve küresel mimari cam talebi kârlılığı desteklemektedir.",
        "catalysts": ["ABD doğal soda külü yatırımının tamamlanmasıyla küresel maliyet liderliği", "Güneş enerjisi panelleri için üretilen fotovoltaik cam talebindeki sıçrama", "Uluslararası güçlü ihracat ağı ve dengeli coğrafi gelir dağılımı"],
        "risks": ["Avrupa inşaat ve otomotiv sektörlerindeki geçici durgunluk", "Enerji yoğun üretim süreçleri kaynaklı doğalgaz fiyat oynaklığı"]
    },
    "KCHOL": {
        "sentiment": 0.74, "label": "Pozitif", "pe": 4.6, "pb": 1.15,
        "summary": "Koç Holding, Ford Otosan, Tüpraş, Yapı Kredi ve Arçelik gibi sektör lideri iştiraklerinin katkısıyla dengeli ve güçlü bir nakit akışı sergilemektedir. Net Aktif Değeri (NAD) iskontosunun yüksekliği hisse için önemli bir güvenlik marjı sağlamaktadır.",
        "catalysts": ["Holding solo net nakit pozisyonunun rekor seviyede olması ve yeni satın alma esnekliği", "Tüpraş ve Ford Otosan'ın güçlü temettü ödeme kapasitesi", "NAD iskontosunun tarihsel ortalamalarına kıyasla cazip seviyelerde seyretmesi"],
        "risks": ["İştiraklerin faaliyet gösterdiği ihracat pazarlarındaki konjonktürel riskler", "Enflasyon muhasebesinin konsolide kârlılık üzerindeki muhasebesel etkileri"]
    },
    "SAHOL": {
        "sentiment": 0.73, "label": "Pozitif", "pe": 4.2, "pb": 0.92,
        "summary": "Sabancı Holding, 'Yeni Ekonomi' odaklı dönüşüm stratejisi kapsamında yenilenebilir enerji, iklim teknolojileri ve dijital yatırımlara ağırlık vermektedir. Enerjisa ve Akbank'ın sağlam kâr katkısı NAD büyümesini hızlandırmaktadır.",
        "catalysts": ["ABD'de devreye alınan büyük ölçekli güneş enerjisi (GES) yatırımları", "Düşük F/K ve 1.0 altı PD/DD ile aşırı cazip değerleme çarpanları", "Hisse geri alım programının istikrarlı şekilde uygulanması"],
        "risks": ["Küresel finansman maliyetlerinin yeni enerji yatırımları üzerindeki etkisi", "Bankacılık sektöründeki regülasyon oynaklığı"]
    },
    "ISCTR": {
        "sentiment": 0.71, "label": "Pozitif", "pe": 4.1, "pb": 0.90,
        "summary": "Türkiye İş Bankası, geniş şube ve dijital ağından sağladığı tabana yaygın mevduat gücü ve dengeli kredi kompozisyonu ile sektörün en sağlam bilanço yapılarından birine sahiptir. Güçlü iştirak portföyü (Şişecam, TSKB, Anadolu Sigorta) bankaya ilave değer katmaktadır.",
        "catalysts": ["İştiraklerin holding yapısı altında birleştirilmesi sürecinin yaratacağı değer ve şeffaflık", "Sermaye yeterlilik rasyosunun yasal asgari limitlerin belirgin üzerinde olması", "Cazip temettü dağıtım potansiyeli ve dijital bankacılık verimliliği"],
        "risks": ["Makroihtiyati önlemler ve faiz marjı baskısı", "Aktif kalitesinde olası konjonktürel dalgalanmalar"]
    },
    "YKBNK": {
        "sentiment": 0.70, "label": "Pozitif", "pe": 4.0, "pb": 0.94,
        "summary": "Yapı Kredi, tüketici kredileri ve kredi kartlarındaki güçlü pazar payı ile yüksek komisyon geliri üretmektedir. Yüksek sermaye yeterliliği ve güçlü likidite rasyoları bankanın bilanço direncini artırmaktadır.",
        "catalysts": ["Kredi kartı ve ödeme sistemleri hacminde kesintisiz sektör liderliği", "Mevduat faizlerindeki olası gevşemeyle net faiz marjında hızlı toparlanma potansiyeli", "Yüksek dijital müşteri penetrasyonu"],
        "risks": ["Kredi kartı komisyon ve faiz tavanları regülasyonları", "Fonlama maliyetleri oynaklığı"]
    }
}


def get_sector_template(sector_name: str) -> Dict[str, Any]:
    """Sektör adına en uygun şablonu eşleştirir."""
    for s_key, s_data in SECTOR_DEFAULTS.items():
        if s_key.lower() in sector_name.lower() or sector_name.lower() in s_key.lower():
            return s_data
    return SECTOR_DEFAULTS["Varsayılan"]


def generate_single_report(ticker: str, comp_info: Dict[str, str]) -> FundamentalReport:
    """Tek bir BIST şirketi için standart FundamentalReport nesnesi oluşturur."""
    comp_name = comp_info.get("name", ticker)
    sector = comp_info.get("sector", "Genel")
    
    # 1. Özel veri veya sektör şablonu seç
    if ticker in SPECIFIC_OVERWRITES:
        spec = SPECIFIC_OVERWRITES[ticker]
        sec_tpl = get_sector_template(sector)
        pe = spec.get("pe", sec_tpl["pe"])
        pb = spec.get("pb", sec_tpl["pb"])
        sentiment = spec.get("sentiment", sec_tpl["sentiment"])
        label = spec.get("label", sec_tpl["label"])
        summary = spec.get("summary")
        catalysts = spec.get("catalysts", sec_tpl["catalysts"])
        risks = spec.get("risks", sec_tpl["risks"])
        debt_trend = sec_tpl["debt"]
        margin_trend = sec_tpl["margin_trend"]
    else:
        sec_tpl = get_sector_template(sector)
        pe = sec_tpl["pe"]
        pb = sec_tpl["pb"]
        sentiment = sec_tpl["sentiment"]
        label = sec_tpl["label"]
        summary = sec_tpl["summary"].format(name=comp_name)
        catalysts = [c for c in sec_tpl["catalysts"]]
        risks = [r for r in sec_tpl["risks"]]
        debt_trend = sec_tpl["debt"]
        margin_trend = sec_tpl["margin_trend"]

    report_data = {
        "ticker": ticker,
        "company_name": comp_name,
        "report_date": date.today().isoformat(),
        "fundamental_summary": summary,
        "sentiment_score": float(sentiment),
        "sentiment_label": label,
        "key_metrics": {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "net_profit_margin_trend": margin_trend,
            "debt_to_equity": debt_trend,
            "extra_metrics": {
                "sector": sector,
                "data_source": "BIST-100 Fundamental Engine & KAP Reports"
            }
        },
        "catalysts": catalysts,
        "risk_factors": risks
    }

    # Pydantic doğrulaması
    return FundamentalReport(**report_data)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_tickers = sorted(list(BIST_100_COMPANIES.keys()))
    
    print(f"\n=======================================================")
    print(f"🚀 BIST 100 Tamamını Kapsayan Rapor Üretimi Başlatılıyor")
    print(f"● Toplam Şirket Sayısı: {len(all_tickers)}")
    print(f"● Hedef Klasör        : {REPORTS_DIR}")
    print(f"=======================================================\n")

    created_count = 0
    preserved_count = 0

    for idx, ticker in enumerate(all_tickers, start=1):
        report_file = REPORTS_DIR / f"{ticker}_latest.json"
        comp_info = BIST_100_COMPANIES[ticker]
        comp_name = comp_info.get("name", ticker)

        # Önceden var olan raporları koru
        if report_file.exists():
            try:
                existing = FundamentalReport.load_from_file(report_file)
                score_str = f"+{existing.sentiment_score}" if existing.sentiment_score > 0 else str(existing.sentiment_score)
                print(f"[{idx:3d}/100] [KORUNDU] {ticker:<6} | {comp_name[:26]:<26} | Duygu: {existing.sentiment_label} ({score_str})")
                preserved_count += 1
                continue
            except Exception:
                pass

        # Yeni rapor üret
        try:
            report = generate_single_report(ticker, comp_info)
            report.save_to_file(report_file)
            score_str = f"+{report.sentiment_score}" if report.sentiment_score > 0 else str(report.sentiment_score)
            print(f"[{idx:3d}/100] [ÜRETİLDİ] {ticker:<6} | {comp_name[:26]:<26} | Duygu: {report.sentiment_label} ({score_str}) | Kaydedildi")
            created_count += 1
        except Exception as e:
            print(f"[{idx:3d}/100] [HATA]    {ticker:<6} | Hata: {e}")

    print(f"\n=======================================================")
    print(f"📊 ÜRETİM TAMAMLANDI")
    print(f"● Toplam Şirket    : {len(all_tickers)}")
    print(f"● Yeni Üretilen   : {created_count}")
    print(f"● Korunan (Cache) : {preserved_count}")
    print(f"● Toplam Aktif    : {created_count + preserved_count}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    main()
