var app = new Vue({
  el: '#app',
  data: {
    step: 1,
    idleList: [{
        "id": 1,
        "title": "程起社书条",
        "image": "https://unsplash.it/70/90?random",
        "description": "气特设务支与管学通西第代少大达。交部整听价准候活消以斯石除式。信高相然入使名量以新位间间人事但。听儿但育现支之军置方山文应利。器他非价教条织在省计周国。于几提至进流其世世程边史府示样据中。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 2,
        "title": "量上头利人",
        "image": "https://unsplash.it/70/90?random",
        "description": "什处手高达铁战作知这做海道。议阶品传加二正属身传议的安。图认原标段精白步历干则心社往名新。织件情铁斗价界构号家议受支外些拉。管规已类响一拉儿众说他马。议口务速难头关眼该好消容度术最明建作。",
        "price": 9,
        "selected": false,
        "mount": 1
      },
      {
        "id": 3,
        "title": "达眼解构王六",
        "image": "https://unsplash.it/70/90?random",
        "description": "快意却名北公只着接然群青说通。器革民等率史状多感没养件那次理参适。太红变例就族建素党进采见物林精满。无给值设品新山京行线会才统高何。决什运空行家机周验些许我百种北。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 4,
        "title": "反六条用般",
        "image": "https://unsplash.it/70/90?random",
        "description": "指科立再类管事会金只化但此无管。到阶成时府无五务业反价受节传。质条儿飞达产使除部说许三把织。只权问己传习质产地专书单们记地四管部。中设片计般容统新连群无近后道三。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 5,
        "title": "青验真所音",
        "image": "https://unsplash.it/70/90?random",
        "description": "路省段量每里看斯步五主变因然至。现率音军己原作传路太飞毛受转段化使。他调给个可府先思后当百高使克以。转常性属第求素空第军改王却给。那中有物车断打应五应和队位。",
        "price": 5,
        "selected": false,
        "mount": 1
      },
      {
        "id": 6,
        "title": "体花前众解并面",
        "image": "https://unsplash.it/70/90?random",
        "description": "物回于书计关眼求书团照能南国而照。民支平机观思节音养商展政运国备些。回龙斯声员周你全何农己听五加。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 7,
        "title": "民那包万战们自",
        "image": "https://unsplash.it/70/90?random",
        "description": "也正周际置任类参花着前市据接验需。县到下改因划目气易便反目更。说较产身然毛存劳象可克切我候。约五需却局百型究物越百则可。",
        "price": 8,
        "selected": false,
        "mount": 1
      },
      {
        "id": 8,
        "title": "光放展次放时品",
        "image": "https://unsplash.it/70/90?random",
        "description": "院王消记深其说其提道格白济任战场。住引由几该入阶部民后期平东率。领导身前场具况边石效标任联这可。技便土本制通军关出始题自。",
        "price": 8,
        "selected": false,
        "mount": 1
      },
      {
        "id": 9,
        "title": "色件压且",
        "image": "https://unsplash.it/70/90?random",
        "description": "一活观建三正学区采五照始点强术进太。通团年反头人作育完起干思院置。治管美化解特先能育们容数到百本。他你再政才置别同该标而学变文率。有当合近给交备族领边水断华。名元常商院积主造线体相资原及报路。样资参全北更布育展么况影至飞构学。",
        "price": 9,
        "selected": false,
        "mount": 1
      },
      {
        "id": 10,
        "title": "来交织适及",
        "image": "https://unsplash.it/70/90?random",
        "description": "一被世体影元人性在离子越力不方。眼高导特立价明集书口山单论包。感色工电南应白你光厂状此样今性布。半路里动以且书眼争精之果验系而写调。清非清相现了育平进集族料叫型风十军生。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 11,
        "title": "研平提员",
        "image": "https://unsplash.it/70/90?random",
        "description": "将月重太也表广次查空日律看民队政你特。或克与难住省被属习局共流小无构。和易利清整住八队们值可长安儿始才边。拉得状平除所西后龙力通工快议物无。实示物层压采期使有传式并太月越应龙。空因何往每体公家公权王标节温马至。",
        "price": 7,
        "selected": false,
        "mount": 1
      },
      {
        "id": 12,
        "title": "周查今每证",
        "image": "https://unsplash.it/70/90?random",
        "description": "整年权需间传查存历切手观达拉。做体数走且及于条何象着发想响包。社元周局生往时科好海变广验他种变。离林局场论发约西该单过几般那数。才速去查些边根原社目更际则前料王。什江展江米月计华育市下选知省样省平。",
        "price": 7,
        "selected": false,
        "mount": 1
      },
      {
        "id": 13,
        "title": "大写压派受",
        "image": "https://unsplash.it/70/90?random",
        "description": "持工划争作手员斗龙青要便米该识管又。决深会容动十易看影产天非六军目结半。速就公气象历且斯证效备克自眼改又。快学周段再劳色数军工实般成八。战圆积各准学自质口走位次美验。决个素方二同期始矿火书电很往亲无行又。",
        "price": 9,
        "selected": false,
        "mount": 1
      },
      {
        "id": 14,
        "title": "元打品志需",
        "image": "https://unsplash.it/70/90?random",
        "description": "干几行第世山设元反文现件育消少。温史表使型他车美事设力现。所利向常运战标线象细际间示山。观影少导活议的油决然就层工布思国调声。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 15,
        "title": "属号党建",
        "image": "https://unsplash.it/70/90?random",
        "description": "清米形将社义再团二题议使能提外。按党酸写九都边代办八前候断更。圆点料段采头则设江为于则办需。本空联热结系场还它那造点交。",
        "price": 9,
        "selected": false,
        "mount": 1
      },
      {
        "id": 16,
        "title": "府它八但了美教",
        "image": "https://unsplash.it/70/90?random",
        "description": "统收务土话以业当可保据克热单相效国位。信联年果至即其片走部七话斗全队。小研图保议他眼分规派层少型放铁想标。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 17,
        "title": "林深般市规委",
        "image": "https://unsplash.it/70/90?random",
        "description": "压队及回酸平己亲响变强人。府白历计根该等问际型节越指线声府已。克史至前却很越达新整离千加至期战。色九难九他干习第最对里战可只关于几拉。团证关色心部京县包次题近光山众容建。深书问断备感速看价原事角。",
        "price": 8,
        "selected": false,
        "mount": 1
      },
      {
        "id": 18,
        "title": "已工例道不热",
        "image": "https://unsplash.it/70/90?random",
        "description": "口成按保达县干它变型别在值历。信育布几果信心日干心理南该原心。格方史线形写支制拉养群开较合次情。改五我走品至则上流般他广说计清。下造四总求点员结且少准意工维物。",
        "price": 6,
        "selected": false,
        "mount": 1
      },
      {
        "id": 19,
        "title": "使把每命平周",
        "image": "https://unsplash.it/70/90?random",
        "description": "原量进按看当者油六亲斯二中集再。器子眼可气向石得速区图候装。几图联地道总年影议调适白。记院况须时类又中传成色观着应根可构文。参了线成外按都收照价本油格发两支。",
        "price": 9,
        "selected": false,
        "mount": 1
      },
      {
        "id": 20,
        "title": "南积具先间",
        "image": "https://unsplash.it/70/90?random",
        "description": "火几四动信教高照直此己身少。所加内七具消果程志这北果间省济到总提。解种成毛层它但产较局经金标产商。队矿她算其达对马事积确也效斗你。",
        "price": 5,
        "selected": false,
        "mount": 1
      }
    ],
    selectedMount: 0,
    selectedIdle: [],
    selectIdStr: [],
    loading:false
  },
  computed: {
    totalPrice: function () {
      let total = 0
      for (let i = 0; i < this.selectedIdle.length; i++) {
        total += this.selectedIdle[i].price * 1
      }
      return total
    }
  },
  created: function() {
    this.loading = true
  },
  mounted() {
    this.loading = false
    // axios.get('https://www.fastmock.site/mock/d81ad7202001983b10683b2a404459d0/idle/idle')
    //   .then((res) => {
    //     this.loading = false;
    //     this.idleList = res.data;
    //   })
    //   .catch((err) => {
    //     console.log(err);
    //   });
  },
  methods: {
    handleCardClick(item) {
      if (item.mount > 0) {
        if (item.selected) {
          item.selected = false;
          let index = this.selectedIdle.indexOf(item)
          this.selectedIdle.splice(index, 1)
          this.selectedMount--;
        } else {
          item.selected = true;
          this.selectedIdle.push(item)
          this.selectedMount++;
        }
      }
    },
    getSelectList() {
      if (this.selectedMount > 0) {
        let selectedIdle = [];
        let selectId = [];
        this.idleList.forEach((item) => {
          if (item.selected) {
            selectedIdle.push(item);
            selectId.push(item.id);
          }
        })
        this.selectedIdle = selectedIdle;
        this.selectIdStr = window.btoa(selectId.join('|'));
        this.step = 2;
        // console.log(this.selectIdStr);
      }
    },
    handleBack() {
      this.step = 1;
    }
  }
})