package main

import (
	"bufio"
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/md5"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math/big"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	fixedWidth  = 160
	fixedHeight = 64
)

var (
	charset = []string{" ", "6", "t", "y", "w", "J", "K", "k", "p", "7", "8", "9", "n", "j", "P", "q", "D", "G", "c", "N", "v", "X", "H", "Y", "5", "0", "h", "R", "f", "r", "4", "d", "A", "E", "M", "l", "V", "m", "a", "F", "s", "i", "z", "U", "g", "x", "u", "o", "3", "Q", "b", "e", "T", "1", "2"}

	sfzList []IDCard
	sfzMu   sync.Mutex
)

type Config struct {
	OnnxLibPath    string `json:"onnx_lib_path"`
	OnnxModelPath  string `json:"onnx_model_path"`
	SfzFilePath    string `json:"sfz_file_path"`
	OutputFilePath string `json:"output_file_path"`
	AccountPrefix  string `json:"account_prefix"`
	PasswordPrefix string `json:"password_prefix"`
	WorkerCount    int    `json:"worker_count"`
}

type IDCard struct{ Number, Name string }

type WorkerOCR struct {
	session *ort.AdvancedSession
	input   *ort.Tensor[float32]
	output  *ort.Tensor[int64]
}

// 随机数生成辅助函数
func randInt(max int) int {
	n, _ := rand.Int(rand.Reader, big.NewInt(int64(max)))
	return int(n.Int64())
}

// 固定的 Android Chrome 请求头
func setHeaders(req *http.Request, referer string) {
	req.Header.Set("User-Agent", "Mozilla/5.0 (Linux; Android 13; SM-S918B Build/TP1A.220624.014) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.6099.130 Mobile Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	req.Header.Set("Accept-Encoding", "gzip, deflate, br")
	req.Header.Set("Cache-Control", "max-age=0")
	req.Header.Set("Connection", "keep-alive")

	if referer != "" {
		req.Header.Set("Referer", referer)
	}
}

// 加密函数
func I4399Encrypt(text string) string {
	pass := []byte("lzYW5qaXVqa")
	salt := make([]byte, 8)
	rand.Read(salt)
	keyIv := make([]byte, 0, 48)
	var prev []byte
	for len(keyIv) < 48 {
		h := md5.New()
		h.Write(prev)
		h.Write(pass)
		h.Write(salt)
		prev = h.Sum(nil)
		keyIv = append(keyIv, prev...)
	}
	block, _ := aes.NewCipher(keyIv[:32])
	padding := 16 - (len(text) % 16)
	padded := append([]byte(text), bytes.Repeat([]byte{byte(padding)}, padding)...)
	encrypted := make([]byte, len(padded))
	cipher.NewCBCEncrypter(block, keyIv[32:48]).CryptBlocks(encrypted, padded)
	return base64.StdEncoding.EncodeToString(append(append([]byte("Salted__"), salt...), encrypted...))
}

// OCR初始化
func initWorkerOCR(modelPath string) *WorkerOCR {
	inShape := ort.NewShape(1, 1, fixedHeight, fixedWidth)
	inTensor, _ := ort.NewTensor(inShape, make([]float32, fixedHeight*fixedWidth))
	outShape := ort.NewShape(1, 20)
	outTensor, _ := ort.NewEmptyTensor[int64](outShape)
	sess, _ := ort.NewAdvancedSession(modelPath, []string{"input1"}, []string{"output"}, []ort.ArbitraryTensor{inTensor}, []ort.ArbitraryTensor{outTensor}, nil)
	return &WorkerOCR{session: sess, input: inTensor, output: outTensor}
}

func (ocr *WorkerOCR) Recognize(imgByte []byte) string {
	img, _, err := image.Decode(bytes.NewReader(imgByte))
	if err != nil {
		return ""
	}
	resized := resize.Resize(fixedWidth, fixedHeight, img, resize.Lanczos2)
	data := ocr.input.GetData()
	bounds := resized.Bounds()
	for y := 0; y < fixedHeight; y++ {
		for x := 0; x < fixedWidth; x++ {
			r, _, _, _ := resized.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			data[y*fixedWidth+x] = float32(uint8(r>>8)) / 255.0
		}
	}
	ocr.session.Run()
	var res []string
	var last int64
	for _, v := range ocr.output.GetData() {
		if v != last && v != 0 && v < int64(len(charset)) {
			res = append(res, charset[v])
		}
		last = v
	}
	return strings.Join(res, "")
}

// 业务逻辑函数
func doRegister(username, password string, ocr *WorkerOCR) (string, error) {
	jar, _ := cookiejar.New(nil)
	client := &http.Client{Jar: jar, Timeout: 12 * time.Second}

	// 1. 获取 SessionID
	regFrame := "https://ptlogin.4399.com/ptlogin/regFrame.do?regMode=reg_normal&appId=www_home"
	req0, _ := http.NewRequest("GET", regFrame, nil)
	setHeaders(req0, "")
	resp0, err := client.Do(req0)
	if err != nil || resp0 == nil {
		return "网络故障", fmt.Errorf("err")
	}
	b0, _ := io.ReadAll(resp0.Body)
	resp0.Body.Close()

	sid := ""
	if m := regexp.MustCompile(`captchaId=(.*?)'`).FindStringSubmatch(string(b0)); len(m) > 1 {
		sid = m[1]
	} else if m2 := regexp.MustCompile(`UniLoginChangPIC\('(.*?)'\)`).FindStringSubmatch(string(b0)); len(m2) > 1 {
		sid = m2[1]
	}

	time.Sleep(time.Duration(100+randInt(300)) * time.Millisecond)

	code := ""
	if sid != "" {
		reqI, _ := http.NewRequest("GET", "https://ptlogin.4399.com/ptlogin/captcha.do?captchaId="+sid, nil)
		setHeaders(reqI, regFrame)
		if respI, err := client.Do(reqI); err == nil && respI != nil {
			img, _ := io.ReadAll(respI.Body)
			respI.Body.Close()
			code = ocr.Recognize(img)
		}
	}

	time.Sleep(time.Duration(50+randInt(200)) * time.Millisecond)

	sfzMu.Lock()
	sfz := sfzList[time.Now().UnixNano()%int64(len(sfzList))]
	sfzMu.Unlock()

	v := url.Values{}
	v.Add("appId", "www_home")
	v.Add("regMode", "reg_normal")
	v.Add("sec", "1")
	v.Add("username", username)
	v.Add("password", I4399Encrypt(password))
	v.Add("passwordveri", I4399Encrypt(password))
	v.Add("realname", I4399Encrypt(sfz.Name))
	v.Add("idcard", I4399Encrypt(sfz.Number))
	v.Add("sessionId", sid)
	v.Add("inputCaptcha", code)
	v.Add("reg_eula_agree", "on")

	reqR, _ := http.NewRequest("POST", "https://ptlogin.4399.com/ptlogin/register.do", strings.NewReader(v.Encode()))
	setHeaders(reqR, regFrame)
	reqR.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	reqR.Header.Set("Origin", "https://ptlogin.4399.com")

	respR, err := client.Do(reqR)
	if err != nil || respR == nil {
		return "网络错误", fmt.Errorf("err")
	}
	br, _ := io.ReadAll(respR.Body)
	respR.Body.Close()
	rt := string(br)

	if strings.Contains(rt, "注册成功") || strings.Contains(rt, "postLoginHandler") {
		return "生产成功", nil
	}
	if m := regexp.MustCompile(`id="Msg"[^>]*>(.*?)</div>`).FindStringSubmatch(rt); len(m) > 1 {
		msg := strings.TrimSpace(regexp.MustCompile(`<[^>]+>`).ReplaceAllString(m[1], ""))
		if strings.Contains(msg, "稍后再试") || strings.Contains(msg, "繁忙") {
			return "IP熔断:" + msg, fmt.Errorf("fusing")
		}
		return msg, fmt.Errorf("api")
	}
	return "未知错误", fmt.Errorf("unk")
}

func randomString(n int, charsetStr string) string {
	b := make([]byte, n)
	for i := range b {
		num, _ := rand.Int(rand.Reader, big.NewInt(int64(len(charsetStr))))
		b[i] = charsetStr[num.Int64()]
	}
	return string(b)
}

// 生成账号
func generateAccount(prefix string) string {
	length := 6 + randInt(5)
	return prefix + randomString(length, "0123456789abcdefghijklmnopqrstuvwxyz")
}

// 生成密码
func generatePassword(prefix string) string {
	length := 6 + randInt(5)
	return prefix + randomString(length, "0123456789abcdefghijklmnopqrstuvwxyz")
}

// 读取配置文件
func loadConfig(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var cfg Config
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// 主函数
func main() {
	// 读取配置
	cfg, err := loadConfig("config.json")
	if err != nil {
		fmt.Printf("读取 config.json 失败: %v\n", err)
		return
	}

	// 读取身份证文件
	f, err := os.Open(cfg.SfzFilePath)
	if err != nil {
		fmt.Printf("无法打开身份证文件 %s: %v\n", cfg.SfzFilePath, err)
		return
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		p := strings.Split(sc.Text(), ":")
		if len(p) == 2 {
			sfzList = append(sfzList, IDCard{Number: p[0], Name: p[1]})
		}
	}

	if len(sfzList) == 0 {
		fmt.Println("身份证文件中没有有效的数据")
		return
	}

	// 初始化 ONNX Runtime
	ort.SetSharedLibraryPath(cfg.OnnxLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		fmt.Printf("ONNX Runtime初始化失败: %v\n", err)
		return
	}
	defer ort.DestroyEnvironment()

	results := make(chan string, 100)

	// 结果写入协程
	go func() {
		out, err := os.OpenFile(cfg.OutputFilePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			fmt.Printf("无法打开输出文件: %v\n", err)
			return
		}
		defer out.Close()
		for msg := range results {
			fmt.Println(msg)
			if strings.Contains(msg, "[成功]") {
				re := regexp.MustCompile(`\x1b\[[0-9;]*m`)
				cleanMsg := re.ReplaceAllString(msg, "")
				parts := strings.Split(cleanMsg, " ")
				if len(parts) >= 2 {
					out.WriteString(parts[1] + "\n")
				}
			}
		}
	}()

	// 启动 Worker
	for i := 0; i < cfg.WorkerCount; i++ {
		go func(id int) {
			ocr := initWorkerOCR(cfg.OnnxModelPath)
			for {
				u := generateAccount(cfg.AccountPrefix)
				p := generatePassword(cfg.PasswordPrefix)

				st, err := doRegister(u, p, ocr)
				if err != nil {
					results <- fmt.Sprintf("\033[31m[失败] %s -> %s\033[0m", u, st)
				} else {
					results <- fmt.Sprintf("\033[32m[成功] %s:%s\033[0m", u, p)
				}
				time.Sleep(time.Duration(200+randInt(1000)) * time.Millisecond)
			}
		}(i)
	}
	select {}
}