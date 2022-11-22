# cancer_gov_targeted_drugs.R

# https://www.cancer.gov/about-cancer/treatment/types/targeted-therapies/targeted-therapies-fact-sheet#what-targeted-therapies-have-been-approved-for-specific-types-of-cancer

all <- "Atezolizumab (Tecentriq), nivolumab (Opdivo), avelumab (Bavencio), pembrolizumab (Keytruda), erdafitinib (Balversa), enfortumab vedotin-ejfv (Padcev), sacituzumab govitecan-hziy (Trodelvy)

Bevacizumab (Avastin), everolimus (Afinitor), belzutifan (Welireg)

Everolimus (Afinitor), tamoxifen (Nolvadex), toremifene (Fareston), trastuzumab (Herceptin), fulvestrant (Faslodex), anastrozole (Arimidex), exemestane (Aromasin), lapatinib (Tykerb), letrozole (Femara), pertuzumab (Perjeta), ado-trastuzumab emtansine (Kadcyla), palbociclib (Ibrance), ribociclib (Kisqali), neratinib maleate (Nerlynx), abemaciclib (Verzenio), olaparib (Lynparza), talazoparib tosylate (Talzenna), alpelisib (Piqray), fam-trastuzumab deruxtecan-nxki (Enhertu), tucatinib (Tukysa), sacituzumab govitecan-hziy (Trodelvy), pertuzumab, trastuzumab, and hyaluronidase-zzxf (Phesgo), pembrolizumab (Keytruda), margetuximab-cmkb (Margenza)

Bevacizumab (Avastin), pembrolizumab (Keytruda)

Cetuximab (Erbitux), panitumumab (Vectibix), bevacizumab (Avastin), ziv-aflibercept (Zaltrap), regorafenib (Stivarga), ramucirumab (Cyramza), nivolumab (Opdivo), ipilimumab (Yervoy), encorafenib (Braftovi), pembrolizumab (Keytruda)

Imatinib mesylate (Gleevec)

Lanreotide acetate (Somatuline Depot), avelumab (Bavencio), lutetium Lu 177-dotatate (Lutathera), iobenguane I 131 (Azedra)

Pembrolizumab (Keytruda), lenvatinib mesylate (Lenvima), dostarlimab-gxly (Jemperli)

Trastuzumab (Herceptin), ramucirumab (Cyramza), pembrolizumab (Keytruda), nivolumab (Opdivo), fam-trastuzumab deruxtecan-nxki (Enhertu)

Cetuximab (Erbitux), pembrolizumab (Keytruda), nivolumab (Opdivo) 

Imatinib mesylate (Gleevec), sunitinib (Sutent), regorafenib (Stivarga), avapritinib (Ayvakit), ripretinib (Qinlock)

Denosumab (Xgeva), pexidartinib hydrochloride (Turalio)

Bevacizumab (Avastin), sorafenib (Nexavar), sunitinib (Sutent), pazopanib (Votrient), temsirolimus (Torisel), everolimus (Afinitor), axitinib (Inlyta), nivolumab (Opdivo), cabozantinib (Cabometyx), lenvatinib mesylate (Lenvima), ipilimumab (Yervoy), pembrolizumab (Keytruda), avelumab (Bavencio), tivozanib hydrochloride (Fotivda), belzutifan (Welireg)

Tretinoin (Vesanoid), imatinib mesylate (Gleevec), dasatinib (Sprycel), nilotinib (Tasigna), bosutinib (Bosulif), rituximab (Rituxan), alemtuzumab (Campath), ofatumumab (Arzerra), obinutuzumab (Gazyva), ibrutinib (Imbruvica), idelalisib (Zydelig), blinatumomab (Blincyto), venetoclax (Venclexta), ponatinib hydrochloride (Iclusig), midostaurin (Rydapt), enasidenib mesylate (Idhifa), inotuzumab ozogamicin (Besponsa), tisagenlecleucel (Kymriah), gemtuzumab ozogamicin (Mylotarg), rituximab and hyaluronidase human (Rituxan Hycela), ivosidenib (Tibsovo), duvelisib (Copiktra), moxetumomab pasudotox-tdfk (Lumoxiti), glasdegib maleate (Daurismo), gilteritinib (Xospata), tagraxofusp-erzs (Elzonris), acalabrutinib (Calquence), avapritinib (Ayvakit), brexucabtagene autoleucel (Tecartus)

Sorafenib (Nexavar), regorafenib (Stivarga), nivolumab (Opdivo), lenvatinib mesylate (Lenvima), pembrolizumab (Keytruda), cabozantinib (Cabometyx), ramucirumab (Cyramza), ipilimumab (Yervoy), pemigatinib (Pemazyre), atezolizumab (Tecentriq), bevacizumab (Avastin), infigratinib phosphate (Truseltiq), ivosidenib (Tibsovo)

Bevacizumab (Avastin), crizotinib (Xalkori), erlotinib (Tarceva), gefitinib (Iressa), afatinib dimaleate (Gilotrif), ceritinib (LDK378/Zykadia), ramucirumab (Cyramza), nivolumab (Opdivo), pembrolizumab (Keytruda), osimertinib (Tagrisso), necitumumab (Portrazza), alectinib (Alecensa), atezolizumab (Tecentriq), brigatinib (Alunbrig), trametinib (Mekinist), dabrafenib (Tafinlar), durvalumab (Imfinzi), dacomitinib (Vizimpro), lorlatinib (Lorbrena), entrectinib (Rozlytrek), capmatinib hydrochloride (Tabrecta), ipilimumab (Yervoy), selpercatinib (Retevmo), pralsetinib (Gavreto), cemiplimab-rwlc (Libtayo), tepotinib hydrochloride (Tepmetko), sotorasib (Lumakras), amivantamab-vmjw (Rybrevant)

Ibritumomab tiuxetan (Zevalin), denileukin diftitox (Ontak), brentuximab vedotin (Adcetris), rituximab (Rituxan), vorinostat (Zolinza), romidepsin (Istodax), bexarotene (Targretin), bortezomib (Velcade), pralatrexate (Folotyn), ibrutinib (Imbruvica), siltuximab (Sylvant), idelalisib (Zydelig), belinostat (Beleodaq), obinutuzumab (Gazyva), nivolumab (Opdivo), pembrolizumab (Keytruda), rituximab and hyaluronidase human (Rituxan Hycela), copanlisib hydrochloride (Aliqopa), axicabtagene ciloleucel (Yescarta), acalabrutinib (Calquence), tisagenlecleucel (Kymriah), venetoclax (Venclexta), mogamulizumab-kpkc (Poteligeo), duvelisib (Copiktra), polatuzumab vedotin-piiq (Polivy), zanubrutinib (Brukinsa), tazemetostat hydrobromide (Tazverik), selinexor (Xpovio), tafasitamab-cxix (Monjuvi), brexucabtagene autoleucel (Tecartus), crizotinib (Xalkori), umbralisib tosylate (Ukoniq), lisocabtagene maraleucel (Breyanzi), loncastuximab tesirine-lpyl (Zynlonta)

Ipilimumab (Yervoy), nivolumab (Opdivo)

Pembrolizumab (Keytruda), dostarlimab-gxly (Jemperli)

Bortezomib (Velcade), carfilzomib (Kyprolis), panobinostat (Farydak), daratumumab (Darzalex), ixazomib citrate (Ninlaro), elotuzumab (Empliciti), selinexor (Xpovio), isatuximab-irfc (Sarclisa), daratumumab and hyaluronidase-fihj (Darzalex Faspro), belantamab mafodotin-blmf (Blenrep), idecabtagene vicleucel (Abecma)

Imatinib mesylate (Gleevec), ruxolitinib phosphate (Jakafi), fedratinib hydrochloride (Inrebic)

Dinutuximab (Unituxin), naxitamab-gqgk (Danyelza)

Bevacizumab (Avastin), olaparib (Lynparza), rucaparib camsylate (Rubraca), niraparib tosylate monohydrate (Zejula)

Erlotinib (Tarceva), everolimus (Afinitor), sunitinib (Sutent), olaparib (Lynparza), belzutifan (Welireg)

Selumetinib sulfate (Koselugo)

Cabazitaxel (Jevtana), enzalutamide (Xtandi), abiraterone acetate (Zytiga), radium 223 dichloride (Xofigo), apalutamide (Erleada), darolutamide (Nubeqa), rucaparib camsylate (Rubraca), olaparib (Lynparza)

Vismodegib (Erivedge), sonidegib (Odomzo), ipilimumab (Yervoy), vemurafenib (Zelboraf), trametinib (Mekinist), dabrafenib (Tafinlar), pembrolizumab (Keytruda), nivolumab (Opdivo), cobimetinib (Cotellic), alitretinoin (Panretin), avelumab (Bavencio), encorafenib (Braftovi), binimetinib (Mektovi), cemiplimab-rwlc (Libtayo), atezolizumab (Tecentriq)         

Pazopanib (Votrient), alitretinoin (Panretin), tazemetostat hydrobromide (Tazverik)

Pembrolizumab (Keytruda)

Larotrectinib sulfate (Vitrakvi), entrectinib (Rozlytrek)

Pembrolizumab (Keytruda), trastuzumab (Herceptin), ramucirumab (Cyramza), fam-trastuzumab deruxtecan-nxki (Enhertu), nivolumab (Opdivo)

Imatinib mesylate (Gleevec), midostaurin (Rydapt), avapritinib (Ayvakit)

Cabozantinib (Cometriq), vandetanib (Caprelsa), sorafenib (Nexavar), lenvatinib mesylate (Lenvima), trametinib (Mekinist), dabrafenib (Tafinlar), selpercatinib (Retevmo), pralsetinib (Gavreto)"

require(stringr)
temp <- lapply(str_extract_all(string=all, pattern = ".+\\s\\(\\w+\\)"), FUN = strsplit, split = ",")
temp <- unlist(temp)
temp <- trimws(temp)

names <- trimws(str_replace(temp, "(.+)\\(.+\\)", replacement = "\\1"))
brand_names <- trimws(str_replace(temp, ".+\\((.+)\\)", replacement = "\\1"))

all_targeted_drugs <- c(names, brand_names)
fwrite(data.table(Targeted_Drugs = all_targeted_drugs), "Data/DRP_Training_Data/CANCER_GOV_TARGETED_DRUGS.csv")
fread("Data/DRP_Training_Data/CANCER_GOV_TARGETED_DRUGS.csv")
