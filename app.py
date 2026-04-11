#!/usr/bin/env python3
# app.py - Login BQL + Autosurf (sequenza identica al vecchio login)

import os
import time
import json
import threading
import gc
import requests
import numpy as np
import cv2
import faiss
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from datasets import load_dataset

# ==================== CONFIGURAZIONE ====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "/tmp/errori"
HEALTH_CHECK_PORT = int(os.environ.get('PORT', 10000))
MAX_CONSECUTIVE_FAILURES = 5

EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"
REFERER_URL = "https://www.easyhits4u.com/?ref=nicolacaporale"
BROWSERLESS_URL = "https://production-sfo.browserless.io/chrome/bql"

# ==================== CHIAVI VALIDE ====================
VALID_KEYS = [
    "2TPBw78eoqITsdsc25e9ff6270092838010c06b1652627c8f",
    "2UB2mJ8Pu4KvAwya658a33c2af825bbe2f707870ba088d746",
    "2UB6xXPVzalwmFrdf68265d93b745fd095899467d21a32326",
    "2UB72G0jNe5RsxL6b2e845d0b94bb6897966e88f662bc99a7",
    "2UCe01EH3vUJLnP6d3f028660d770ed840a0c6b05b6dcf71e",
    "2UCyusO830dLAcyda29244c83c2bfa0217728908ff8810c42",
    "2UD3pQCcge39YhQce5797773c8508515a295a1298d0105b42",
    "2UDOf1dHJeNmeOl0a373211ade4280ba7e212cde93dfc9e20",
    "2UDOnpiBIFokFEBcb1017abfdd901756272f2ff182c4a9f32",
    "2UDPWeUf62vB2I8aa37152a5b515e5360c127d669b813f23c",
    "2UG2TlpDxsQJn2Wd1f204756127d4ac2136b41bd01baaa0ca",
    "2UGdbQnmFCJwS9Vd714eb85438cf63d00a8f878a898cfe865",
    "2UGdcalCbtmQNCt0c0a65e134b1833ed5d77b0c27fec4df7a",
    "2UGdeyvPnuYf2tm78f5d97e862f004feef3a8e41dfd58b3ef",
    "2UGdfrLYfztPfpy65ea1648786cdfe855a89073f49a24fa15",
    "2UGdh0XeC72wcccb12714bdae43194a6a8647ce9a836d9cf9",
    "2UGdiXdiszEa5rw5c83ff671b0f30e6b45cb159d1b7a8f221",
    "2UH1q8Mnj1ERdcZf243e8d19a8e05da8998570d64e212cc3a",
    "2UH1rvpwwnyIqKYf3d2b847c23f1bf100eb78217b4abe399e",
    "2UH1tCPjVWSuutr98a6d9529fb8c03b457496afe6466ebac0",
    "2UH1uDTJQKxWMi750e2ad5d114a378275b4f4963b81476824",
    "2UH1xtruDYkpN6qafcf735210a0d390f38b7934fee7020509",
    "2UH1yEsOSdMyVgBb79e5d9f7283da3ab24b099772a221c0c1",
    "2UH200RyjgTPJAyd69e6979481a42076d9715120add383b2f",
    "2UH21NyLelnPOXN89ef213e06c030d3a20fe91f74ed023cd6",
    "2UH23g4Tjer24qYda1b38b3bf4995babae59f6ade1b5d80d5",
    "2UH24rd152tYgA9bfd616f9e0a1eee38c91957e77f7388367",
    "2UH26buZuikxxt088fe658690e962e79f00f03bae1c9c23d3",
    "2UH27IyTT0RHycacd91e7dcd3c026b13a34334e2669771ff3",
    "2UH294cqCAfyXPYa0fb233ea57a4aa4ac1cfa9e767080324b",
    "2UH2AnTc77FXlItd61132c9805d95deacff876085a8673a9f",
    "2UH2CfWXJrCUNeVdd80c7e1b03518bbfdcf651e646f5f87d6",
    "2UH2DCjQeXY976cbc3b9a2f96b6b7c639bce3f82349f4dc3c",
    "2UH2FdGsdqj9zdBd31de95f2d5f8f661cf0cd4980112ce6d5",
    "2UH2GTfPxLjEANac954251257e3745ed64d7eeba896e59569",
    "2UH2IvxBVMIZf7pbc1f54a2696deef605bc9a8b43b5ccc8b8",
    "2UH2JmJbYEUBMQBa05981954be8f4996b345b0f8b3682cc00",
    "2UH2L4xZ5oNQ80w85bf6bc0075e1f1e91f9106ad882b73ad3",
    "2UH2N0WXkIuziiJ071449dfda09a57c174a3271491197bc93",
    "2UH2PXIv41CFGZi83f01bc2ec164655754bffb8a14e6ec8dd",
    "2UH2QtMa2NAHKqgff261a53ca86a8f8281fc78b3d18d61829",
    "2UH2bnJSP3jJh2zcfddf0eaacc03a5a36a586558c9127f6a0",
    "2UH2cb7PyfPpoxBce3f0a9868715cd7026d8e539aac36d402",
    "2UH2eKbGQKuIYUXcad0304cb6e5bee0b0c403afdbb45eb29e",
    "2UH2gUfSx5xbV8v5c1782e505ebd7c097193963887490ccf2",
    "2UH2hBN40tQzuef302dcb8aa91dbe6770856a538edbfb6673",
    "2UJ1tyeQPpoIq5ce393e8aedfc71bf2cd5bde8e12ce0840b8",
    "2UJ1uodfUujigTy9334dd24560921ca34118bd518d88ab3de",
    "2UJ1w5OGJhvYl9K157df6861598fc12388573b068a1a7894e",
    "2UJ1xHDfiYWz8n128afde5f8c9b0da82d7b3e9bdfaddedfb4",
    "2UJ1z8AHhwdXTtFdb03f29009a1cfede61bdfb76c90d22468",
    "2UJ2075CjdEW1XH6d65b6e14eebb068e25f42e44b4d292f7d",
    "2UJ22dGBJPN8Thf90e088d29b5553527a52c6f439a20be5f8",
    "2UJ24oJaxnpnLqd1552d3b745a167815b852ad7ba4178fd9a",
    "2UJ26RXsOPYD7rjd1da226f512e5c907bcf92d7ac515944a9",
    "2UJ278vHM8JseeE61c297cafdc30dea2417a01b7eeeeffe1a",
    "2UJK3J6z8WVUZCnebd8f5f45581cb8e33d54c5f102ff1ca1a",
    "2UJK4Jun2RJGbpmb4744ac717d57e27d86a6f8cdea79ecb29",
    "2UJK6yKb6025jjV0ec93e78221afdd7422cba5e9c2cf215b2",
    "2UJK7NrAnPQHmLj5f59ebaeb40664e36acb5e9edb16258649",
    "2UJK9tJF6fSUI1yee47518cbceb44b754091f65ffb37385e9",
    "2UJKBhoRgHclEJteebf8c7771d9b2ac024e173d5e8c668e63",
    "2UJKIOxgYKTLcPm78093f2ec30b29d3ed2796fd80812e30e4",
    "2UJKKhepmZphDJ5934eae8e34c8cc2166d53c97e18d88842f",
    "2UJKQznXMrCRsDe7e27bb3392684dc84617e99bfebb86c6f3",
    "2UJKSaJ8LbR6yPMa1de82874dc44ebb02c6538905563345db",
    "2UJKT2s7366Q95C9f93fe45d1e69c35b063479f681746371d",
    "2UJKVmGU7EHnhBa7888f73274495565fd975f87911d955624",
    "2UJKWATu0ywwDLj6745bd019eb949bc89ee0bde7b8aefcceb",
    "2UJKYEO1MHAoKLP13c6d573801f1194a2db77382e1c9ca279",
    "2UJKZKYxDW04Fvycd08e1f4373d86ac84939fd2da94b7bb6b",
    "2UJKbrJ8mu81DDId0b0a6d5d6f09d4232e86c95d0508d2286",
    "2UJKctR3YKrr0jNb8125ac5a469ffc015154b0ef2ebdfbd64",
    "2UJKgrHd3wpB8ER8967264f75287a7a37b6c07cd1aa385e8a",
    "2UJKhZ4nnu4dmrJdf0abe5a76fdcaccca3d4bde1c8e756207",
    "2UJKj90nSPoPzbGb83f67c8d51804f74d9c294296731f16d2",
    "2UJKkfMpBrzmSiT79baea21763312b842e8d76f0294e4922b",
    "2UJKmcIjXmvO8XRd305c7cbfdc378b8c2b51b9a01431bdb05",
    "2UJKneJJJ1geO9fc62ab067a9bba69951fc680ac31f68b318",
    "2UJKpDVy5uUTm5ke53ea97a630f3e9d40890cda9536bef640",
    "2UJKqHrcH8eFPru090c449d6f3349f227d34e0b64e05d9515",
    "2UJKstbtf4Yhhihbee152f1524f4b30b4169646dacc6f57b9",
    "2UJKu8uOav7pPEid44f93cbe5506e74a6aabdb11f8c3c51bb",
    "2UJKwZYaR9WoIdNdef8ca3188b79f0f1642cf040f908a6f0d",
    "2UJL4KwP022lo430841ccf7077da9eddca8f0e600bbb78ed0",
    "2UJL5zV86C4xbIwe2fa97d07bb917acf5c449b96ab73c9241",
    "2UJLTjMfcMlNuaw76c40208b2a61ce7f8fffa3fd8b570f8be",
    "2UJeHRc4usgkqaz79290cfc0a590e7e510d2f08e3c1e097a7",
    "2UJeImDLl4M1QVL79e96d5c02514a1b5e0157496cf6ac9475",
    "2UJeKKcnl4LdIsD4ec6687797b4e2cea063912c0d4967bf20",
    "2UJeLqrIhFlwcAL44e8c4a5eaaade7a7e19e9d30d9ecf745f",
    "2UJeNUpd130LY7ye54856c2ff7a97031c559923416b6cb1eb",
    "2UJeOj9dDNZwQSe44c0213a252219b135423deea462df3742",
    "2UJeQ20fi76QfZMc9868108ddcf0d48f7d222e01f84f06160",
    "2UJeR8VbtNi89Vt2b11a7f55bfa2b2ef72962a4c4c79e7b4e",
    "2UJeTwUPmdNy1U3fe6f1e133aa046ac533a98381fa6002741",
    "2UJeUDEUPfpMEazcb8f9432da824f8385b44cc4f0ebf76f8e",
    "2UJeWhojOOtI9ua960eced70da633194069a88075c84a4247",
    "2UJeXt7vLUJPzTx65877a3b16d9c3427667455d434f5cbe01",
    "2UJeZQRLQsc4Ty30f80b36dc777f88f874a26004f76c0e1bc",
    "2UJeaoeGNGzIllu26db8f2958c83977df74828bc789642d7d",
    "2UJecquayHuWg0L0a2f9caeb8445169a845e72086c51f78dd",
    "2UJeffW9m0QioGw1bf5ab6701f2f7ddd981e0ba0b439fdf0f",
    "2UJegB5Llj81pb85329d1e0b22ec9544c9d1c6271e51baecd",
    "2UJeiO4ViZxmB9Q3a4cfce6169f082345f99e9fb81cdc0171",
    "2UJejySi3n9QuOf7b618c271a8553ef4e3c908ca05d5b9d04",
    "2UJelBVclIMxMXDec0ccee7c9c9cfdf2bb3f6dbc602360bba",
    "2UJemDosXsRVjY919899c5a5baf50ceefd40b4d3eccc0888d",
    "2UJeoT1lQEHT54J58df61c91743194dc07f54172f0364b49f",
    "2UJep3iSp6BRMhY10e5b1e2a301eac0a4ab83fc0cd95f2daa",
    "2UJereqJb073ZjO7bfb9c4144af6b914e771ea406cdb35abe",
    "2UJesHnbT5R6Qm363d71f8cd44e20054a439e08dfaf00c863",
    "2UJeu8FPmt93u6re3073468d5b20e75bec81d0c3bdd954fcd",
    "2UJevSWhRHZU8yPebc9b765abf1211bf5be17a35034cdd5dd",
    "2UJexmYA5TT1lelaaeed0c1375a753b5ece0abdc288bfc579",
    "2UJez1NeZo9IANC39af39db97d8dc510eada079e67a382cac",
    "2UJf5Y7XOIEr1C0a8d164429e131c03791310f5c730ebda0f",
    "2UJf63gonN7nzTS9ea1f719a0ce5927581327c092de1256fd",
    "2UJf8yCXIpTslbZ33b7286cd0035d86a2d0cf4805aff05fd2",
    "2UJf9j4gNbXkJ0Df9fbd26fe2b7bbddf936ce1291ee79e757",
    "2UJfBYHSunBPyyNcea57b9851512b4634a49a75eeb3fd2a49",
    "2UJfDYGnbPWxoRrf52db3b0cdaaa95105c1508d5c95a94279",
    "2UJfEQuovfxKeIyf6a0bceefd2daa367a9aa8441af1cb1600",
    "2UJfGErgsvVok2956a38deb7fdb710c0cfb8d0cf87ecf1f47",
    "2UJfJSRVsmPBtjnc6827a170c14b610f25acad5a83a85cb79",
    "2UJfLM7LoqVceLCc2bcc79578632f34268b5f5822df51d228",
    "2UJfM4eNfRa4vLq6c270b9edd2090475cc0e2d2c91a0b7db6",
    "2UJfOeucJnuJwbVd60d930923beca50b4f37453876956220f",
    "2UJfPTDe2FdwS3I7f4a18d0e1349bfd4fdc09b84cfffd2765",
    "2UJfRrjfOA5Enwcc0daf048941098fac5c9167f54cb2aa36c",
    "2UJfUko8Ai23SPIfd7242cfabd1e2e3dc7047333fdfc5a5bc",
    "2UJfVijGcFzFywA08f9a398623e5adfa52aabd9d65481606a",
    "2UJfXTOrNo1oANO6fdaa4054b011c4067c5994136a5857560",
    "2UJfZszQ5yIuj1Ldffd5e90c3925e40ec1ed3e7b613aa68a8",
    "2UJfa9OJbtwLAbad78ff3ebf6f586060a142b7209e6da5bd2",
    "2UJfcRrxm5hPX182c70eb6706e500d29001025240397ee501",
    "2UJfegwDn0WLbX0a6f88668d01996b368010a17c093453b22",
    "2UJffc2JYlXTGAId2bba985735cf88dfe4a36ef2184301f82",
    "2UJfhVKHGweK2R18e7e56a606bfbb289e01e7c79e4ea6a57f",
    "2UJfivd1THn3RiW7443c10943a05437e1fb644a71752d7792",
    "2UJfk8Vwyy2DNyv98ea89b8d2b2271f87e63492dd932a3039",
    "2UJflGiKWg6qkoGc09bfd84140a0cb0793c92a3ffe4e9ae32",
    "2UJfnc5nCIk0gePb1d610241ae0134413f0ffc5b2887fe137",
    "2UJfoaUGm3nz0pO29a517a332572ab10efea4bc26f870ab28",
    "2UJfwO8xZgsGMKr98a01f1b877bc91018a38bc354035ac8e2",
    "2UJfx9YbS0Y6cKE3ce8ded8db321ff617ef5c5c80b76106bb",
    "2UJfzSVaDTBKHcA43a7765c1e6a80807249badb2a666d2d40",
    "2UJg0LAE0JRZpZtf96212df7df32b2275d6d0788fcac0ec1e",
    "2UJg2MHnMyalkzCeb24b4e441a33a5bf790261bd8ebcec853",
    "2UJg31QONRDq4ezace122d1b2164d8ed3abfa2ed0f04a5d4b",
    "2UJg5hOZiewB6aS5f298e97721da9d86e618a10e2637e2fa4",
    "2UJg6WSCcwOWfVwa72c444cd2b4089e4a394a275baee07cb6",
    "2UJg8ddUU5vBKyHbec0341acb1eaf06525de420048928634a",
    "2UJg9p6GCvcE0PD95a901f7e05583e3c354d31c7c3e782516",
    "2UJgBRx1cXAB1Ti0044a6b946b8308cd94e972533009195a2",
    "2UJgDl2vIVOFx2kea4d701641715814c797736e000d8792dc",
    "2UJgEBYoKdSWDAQ4d09fde95cb8ad7c762fbe19de3e85f650",
    "2UJgGKPdpHOkKOs980360092136339839244f766ec8eb0cd1",
    "2UJgI8N6w0UElT2f13d7bf8056d21b21706e6343150451d4e",
    "2UJgJduZYWlWeC93cb85209d4bb0e934fff7fff2641d37ef5",
    "2UJgL7gflMyYpU6ba84489b1b4bf6fbb54253781ecf2be549",
    "2UJgMgvLR36VqKMd540732a6bcf07a675d27ec09693eeeab7",
    "2UJgOOuwtSTW44bab341065b174c4a0ab4bd29552dbe44792",
    "2UJgPlTWEx8V9LCb9b02d45caeca47808fe7f60ae7c6c3a95",
    "2UJgRlv4AdLUoEJd2e0b3de75287df093f59ae9c1cf4882a2",
    "2UJgSZST6QJBH7j2c0223885d0bb7f71a56fac99867096860",
    "2UJgUAVeakOhlXA67241fa6472526de8612927dcd4cae94bd",
    "2UJgVgplNRTe2OP503d8ae8aa40bd2e19e561f29534009679",
    "2UJgXlWaxqpO0XR4407fd3078a651dab5a633597872a272d8",
    "2UJgYbF0pICWMH97e4978155cb33deefd7736aba68a5fdbd7",
    "2UJgafq7FQMQP8b000c9e7d527497c4e380559027f9f19d27",
    "2UJgbSPmkjZ8Ftxcda20c3a970490bc4cd77c86c2e6c2665f",
    "2UJgdrxsny4TNaJe63247af29db0709cd0ce84e2effba1d72",
    "2UJgetmJn8GdVCnca0fae27cd83412ce9269ad894963c5c2e",
    "2UJh5PsFBDC1vyP60997d9cf10ae8769b5530dd715dcd774b",
    "2UJh61mWuDS8O7p9683d48c6acc288cc4d7fcb50a6726551b",
    "2UJh83T7aRvaBYA2bcc82f19c06d20d77319a43e13d9caa77",
    "2UJh9a4ZALgUd7W376b9a7bbbeef6b0eabc4e7239fd8bc19b",
    "2UJhBPBjBPytE5Lf518a5fe22fdc4166aa057fe94e87630e5",
    "2UJhCxG11eBO4mab012bb7dccb9e346e5fd4959c61b1a12b8",
    "2UJhECLFiJlsbMCc1f333233a99f38824bdd9432fa91d9e08",
    "2UJhFdiHbkKvW363a9dd6d2af92ebc925419dca2235901ca4",
    "2UJhHNN5TGnKETJdc5fcd08f2f505055a0fb2e6a8fd9ff535",
    "2UJhIIeSjNkxosi77fc195b190abb75bae52c683585ec9814",
    "2UJhKFjdXvW16O7063bbb6c987360d4b7f261dad43beb833d",
    "2UJhLBN0sBE3E0na6ffeb309d66fb3b9b802fce87c882dd31",
    "2UJhNxaPKBaY8S4738548923636947dd6f1ba07e59fc59bb1",
    "2UJhOJEIAyG4AOwfe5dca8f62865726b9d5a7679f2c8f3aa6",
    "2UJhQ5bQrcIzMsE11eed0a634bc83ccb01c299b716c4d755e",
    "2UJhSvsnCzUUVQS68a35b33c1773cbaf77cd1e0c545e28975",
    "2UJhTbq3KQ4ec9D724e034d38becf708bdc3388c0e8e5dfae",
    "2UJhV1xXBh38C7g193fcdeec5ae425c98f65301cd7291debd",
    "2UJhXqWr1gn8j4t8408030dec802e75cac7179437be6740c7",
    "2UJhY7T9z1PzCKr5af4c8e91c617a0917ac30d09eaae8920e",
    "2UJhakIsjhAObxFfb9af92b6092bb4c43e2de56f220b6969e",
    "2UJhbu3oVfvAlmPbcd98b22b308c251fdb60c964c083b5e20",
    "2UJhdcFCDWSywSr5ab7bd26c0052199d2c88ae430a326a2da",
    "2UJhef1fSMRgJPi3bc0fb4e0b9ced9b02192f72950f048263",
    "2UJdYortHMjTNiwa383d56b609fd32076920cee2497198c72",
    "2UJdZPf4MhGHiynfbdf8e43fdbd955a8e2fd21211ec168a83",
    "2UJdb6qAIyZsGc0a231a9cbe5b4b2a71c7b97dcfc39006610",
    "2UJdcZdTQB12vwmbdc0373cdf1859901abc2b15b73c35aa5c",
    "2UJdeh3jnzMEZrpea460032b96eea0cb17cdc42a27bf43e68",
    "2UJdfAn38waenB16e0fcdcdedcb5b1307ba97d8a6af618a9f",
    "2UJdhgXjYmHbbstd49eebdac15a025628f639b983109cca53",
    "2UJdij0QU3oHXl310b7edcb97d2a863edeb06731ffbe02cda",
    "2UJdkC3uLFUMSTb40350f6c94469a27dda114438ce3e63d61",
    "2UJdllCVXL1KPTO438c158d998f79ee23d6cf846e32f538d2",
    "2UJdnx3LfEKv1AS1c98ec6e53d5177a5dfff1ce2f4fd5c26e",
    "2UJdoDOrc8iB9oY3c8987bd9dd6b57d56c3ffcd129691aa2e",
    "2UJdqTOYZiNUMqJ704533533bf04ad95e5f4b2d7db7bd7aef",
    "2UJdrPPh22e4CsB0b374d7c85e00103591b4ce31e8b4399ff",
    "2UJdtvsfGUXjoCVa06935e1c6f8aca2c90a47977756e48926",
    "2UJduO9zZ6xXgB4fc9634436f57b79b397a93127299e24a66",
    "2UJdwo5Zj5kClzFe2295939a26cdacaf96688e020fce28e3d",
    "2UJdxyxgMS9r7rf03d4a2ce509fa04fc779abdf45938c5e3c",
    "2UJdzCxhc6pyPLY8be16983e7696b06122fce30ac19440aa3",
    "2UJe12p091H5YEN3341f039e09bf3f36727e4e6cb6b45f4fe",
    "2UJe2aJkFkisdwnca8af2c4b2bcb59432a24aa235f3230d07",
    "2UJe4BnPpFBog8X48a7d6349341ec168d94d6ea7d9ae58b35",
    "2UJe66aYhFKXQ1Y3904dc283e5e6511ca8ca657ada0093a0c",
    "2UJe7C0ZodaMsyrc92eb922a8d2e340bcfe66e28d92114d33",
    "2UJe91pN3w0Zm5ef5f7a6ab533bd2cb8a8d9ac6d83d636dd8",
    "2UJeAoeUjUIg3zKdff819c41c5b147249eef5e887902ccea0"
]

# ==================== GLOBALS DATASET ====================
dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33
server_ready = False

# ==================== HEALTH CHECK SERVER =====================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass

def run_health_server():
    global server_ready
    try:
        server = HTTPServer(('0.0.0.0', HEALTH_CHECK_PORT), HealthHandler)
        server_ready = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏥 Health check server avviato sulla porta {HEALTH_CHECK_PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERRORE health check: {e}")
        server_ready = False

health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()
timeout = 10
while not server_ready and timeout > 0:
    time.sleep(0.5)
    timeout -= 0.5

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==================== DATASET HUGGING FACE ====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset da Hugging Face Hub...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=None)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        
        log("🔧 Costruzione indice FAISS (FlatL2) incrementale...")
        index = faiss.IndexFlatL2(vector_dim)
        batch_size = 5000
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:i+batch_size]
            X_batch = np.array(batch['X'], dtype=np.float32)
            index.add(X_batch)
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore dataset: {e}")
        return False

# ==================== FUNZIONI DI FEATURE EXTRACTION ====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])
    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])
    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop).astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(features, 1)
    best_idx = indices[0][0]
    true_label_idx = dataset['y'][best_idx]
    return classes_fast.get(int(true_label_idx), "errore")

def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def salva_errore(qpic, img, picmap, labels, chosen_idx, motivo, urlid=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(ERRORI_DIR, f"{timestamp}_{qpic}")
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, "full.jpg")
    cv2.imwrite(full_path, img)
    for i, p in enumerate(picmap):
        crop = crop_safe(img, p.get("coords", ""))
        if crop is not None and crop.size > 0:
            cv2.imwrite(os.path.join(folder, f"crop_{i+1}.jpg"), crop)
    metadata = {
        "timestamp": timestamp,
        "qpic": qpic,
        "urlid": urlid,
        "motivo": motivo,
        "labels_predette": labels,
        "chosen_idx": chosen_idx,
        "picmap_values": [p.get("value") for p in picmap] if picmap else []
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"📁 Errore salvato in {folder}")

# ==================== LOGIN (esattamente come nel vecchio repo) ====================
def get_cf_token(api_key):
    query = """
    mutation {
      goto(url: "https://www.easyhits4u.com/logon/", waitUntil: networkIdle, timeout: 60000) {
        status
      }
      solve(type: cloudflare, timeout: 60000) {
        solved
        token
        time
      }
    }
    """
    url = f"{BROWSERLESS_URL}?token={api_key}"
    try:
        start = time.time()
        response = requests.post(url, json={"query": query}, headers={"Content-Type": "application/json"}, timeout=120)
        if response.status_code != 200:
            return None
        data = response.json()
        if "errors" in data:
            return None
        solve_info = data.get("data", {}).get("solve", {})
        if solve_info.get("solved"):
            token = solve_info.get("token")
            log(f"   ✅ Token ({time.time()-start:.1f}s)")
            return token
        return None
    except Exception as e:
        log(f"   ❌ Errore token: {e}")
        return None

def do_login(api_key):
    """Esegue login ESATTAMENTE come nel vecchio repository, senza GET extra"""
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/148.0',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': REFERER_URL,
    }
    
    # GET iniziale per cookie (come nel vecchio repo)
    session.get(REFERER_URL, headers=headers, verify=False, timeout=15)
    time.sleep(1)
    
    token = get_cf_token(api_key)
    if not token:
        return None
    
    data = {
        'manual': '1',
        'fb_id': '',
        'fb_token': '',
        'google_code': '',
        'username': EASYHITS_EMAIL,
        'password': EASYHITS_PASSWORD,
        'cf-turnstile-response': token,
    }
    resp = session.post("https://www.easyhits4u.com/logon/", data=data, headers=headers, allow_redirects=True, timeout=30)
    # Attendi che i cookie vengano impostati
    time.sleep(2)
    
    cookies = session.cookies.get_dict()
    if 'user_id' in cookies and 'sesids' in cookies:
        log(f"   ✅ Login OK! user_id={cookies['user_id']}, sesids={cookies['sesids']}")
        # Forza surftype=2 se presente, altrimenti aggiungilo
        if 'surftype' in cookies:
            cookies['surftype'] = '2'
        else:
            cookies['surftype'] = '2'
        for k, v in cookies.items():
            session.cookies.set(k, v)
        # Aggiungi header per AJAX
        session.headers.update({
            'Referer': 'https://www.easyhits4u.com/surf/',
            'X-Requested-With': 'XMLHttpRequest'
        })
        return session
    else:
        log(f"   ❌ Cookie mancanti: {cookies}")
        return None

# ==================== SURF LOOP ====================
def surf_loop(session):
    consecutive_failures = 0
    captcha_counter = 0

    # Opzionale: se il primo POST fallisce, prova a ottenere CSRF token
    need_csrf = False

    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"⚠️ Status {r.status_code}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            data = r.json()
            if data.get("redirect"):
                log(f"⚠️ Sessione scaduta (redirect a {data['redirect']})")
                # Se otteniamo redirect e non abbiamo ancora provato a ottenere CSRF, tentiamo
                if not need_csrf:
                    log("   🔄 Tentativo di ottenere CSRF token...")
                    surf_page = session.get("https://www.easyhits4u.com/surf/", verify=False, timeout=15)
                    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', surf_page.text)
                    if match:
                        csrf = match.group(1)
                        session.headers.update({'X-CSRF-Token': csrf})
                        log(f"   ✅ CSRF token ottenuto: {csrf[:10]}...")
                        need_csrf = True
                        continue  # riprova subito
                break

            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])

            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("⚠️ Dati incompleti, riprovo tra 10 secondi")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(10)
                continue

            consecutive_failures = 0

            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = [predict(c) for c in crops]
            log(f"Labels: {labels}")

            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i

            if chosen_idx is None:
                log("❌ Nessun duplicato trovato")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            resp_json = resp.json()
            if resp_json.get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            captcha_counter += 1
            log(f"✅ OK - indice {chosen_idx} - Totale: {captcha_counter}")
            if captcha_counter % 10 == 0:
                gc.collect()
                log("🧹 Garbage collection")
            time.sleep(2)

        except Exception as e:
            log(f"❌ Eccezione: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
            time.sleep(10)

    log("🏁 Surf loop terminato")

# ==================== MAIN ====================
def main():
    log("=" * 50)
    log("🚀 LOGIN + AUTOSURF (sequenza identica al vecchio login)")
    log("=" * 50)
    
    if not load_dataset_hf():
        log("❌ Dataset non caricato")
        return
    
    for api_key in VALID_KEYS:
        log(f"🔑 Tentativo con chiave: {api_key[:10]}...")
        session = do_login(api_key)
        if session:
            log("✅ Login riuscito! Avvio surf loop...")
            surf_loop(session)
            log("🔄 Surf loop terminato, provo altra chiave...")
        else:
            log(f"   ❌ Login fallito, passo alla prossima chiave")
    
    log("❌ Nessuna chiave funzionante")

if __name__ == "__main__":
    main()
