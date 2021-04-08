let vm = new Vue({
  el: '#app',
  data: {
    visible: false,
    replys: [{
        id: 1,
        name: '审批编号',
        content: '202011080115',
        type: 1
      },
      {
        id: 2,
        name: '提交时间',
        content: '2020/11/8 16:45',
        type: 1
      },
      {
        id: 3,
        name: '所在部门',
        content: '193班',
        type: 1
      },
      {
        id: 4,
        name: '请假类型',
        content: '事假',
        type: 1
      },
      {
        id: 5,
        name: '请假出校门时间',
        content: '2020/11/8',
        type: 1
      },
      {
        id: 6,
        name: '请假具体内容',
        content: '有事外出',
        type: 1
      },
      {
        id: 7,
        name: '当前所在位置',
        content: '',
        type: 2
      },
      {
        id: 8,
        name: '是否明确返校时间是当日21:00前?',
        content: '是',
        type: 1
      },
    ],
    submitTime: '',
    approvalTime: '',
    currentTime: new Date().toLocaleString(),
    qingjia: 1,
    grade: '',
    content: '',
    name: ''
  },
  filters: {
    fmt(val) {
      return val.slice(11,16)
    }
  },
  mounted() {
    // if (this.timer) {
    //   clearInterval(this.timer)
    // }
    // this.timer = setInterval(() => {
    //   this.currentTime = new Date(+new Date() + 8 * 3600 * 1000).toJSON().substr(0, 16).replace("T", " ")
    // }, 1000)
  },
  methods: {
    getHtml() {
      this._formatTime()
      this.visible = false
    },
    hideHtml() {
      this.visible = false
    },
    _formatTime() {
      const prev = new Date(+new Date() + 5 * 2700 * 1000).toJSON().substr(0, 16).replace("T", " ").replace(/-0|-/g, '/')
      const now = new Date(+new Date() + 8 * 3600 * 1000).toJSON().substr(0, 16).replace("T", " ")
      this.submitTime = prev
      this.approvalTime = now.substr(5, ).replace('-', '/')
      let num = this._random()
      this.replys[0].content = now.substr(0, 10).replace(/-/g, '') + '0' + num // 审批编号
      this.replys[1].content = this.submitTime // 提交时间
      this.replys[4].content = now.substr(0, 10).replace(/-0|-/g, '/') // 出校门时间
      this.replys[5].content = this.content
      this.replys[2].content = this.grade
      this.replys[3].content = this.qingjia === 1 ? '事假' : '病假'
    },
    _random() {
      return Math.random() * 200 + 100 | 0
    }
  }
})