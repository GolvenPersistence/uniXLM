#include "gen3rt/quantize_fast.h"
#include <random>
#include <cstdio>
#include <cstring>
#include <vector>
int main() {
    std::mt19937 rng(1234); uint64_t blocks=0;
    auto check=[&](const std::vector<uint32_t>& bits) {
        std::vector<uint8_t> c,s,rc,rs;
        g3rt::quantize_fast(bits.data(),1,int(bits.size()),c,s);
        g3rt::quantize(bits.data(),1,int(bits.size()),rc,rs);
        if(c!=rc || s!=rs) {
            for(size_t i=0;i<bits.size();++i) if(c[i]!=rc[i] || s[i/32]!=rs[i/32]) {
                std::fprintf(stderr,"mismatch bits=%08x code=%u ref=%u scale=%u ref_scale=%u\n",bits[i],c[i],rc[i],s[i/32],rs[i/32]);break;
            }
            return false;
        }
        blocks+=bits.size()/32;return true;
    };
    std::vector<uint32_t> bits(32768);
    // Arbitrary IEEE bits, typical finite inputs, shared exponents, and signed zero.
    for(int round=0;round<256;++round) {
        for(auto& v:bits) {
            v=rng();
            if(round%4==1)v=(v&0x807fffff)|((1+rng()%254)<<23);
            if(round%4==2)v=(v&0x807fffff)|((115+round%20)<<23);
            if(round%4==3){float f=(int(rng()%20001)-10000)/1000.0f;std::memcpy(&v,&f,4);}
        }
        if(!check(bits))return 1;
    }
    // RNE boundaries across the FP32 exponent range, normal/subnormal E4M3,
    // saturation, both signs, and fractions adjacent to each half-way point.
    bits.resize(32);
    for(unsigned e=1;e<255;++e) for(int delta=-30;delta<=15;++delta) for(unsigned mant=0;mant<8;++mant) {
        for(unsigned j=0;j<32;++j) {
            const unsigned se=unsigned(std::max(1,std::min(254,int(e)+delta)));
            const unsigned frac=mant*0x100000+0x80000+(int(j%3)-1);
            bits[j]=(j%2?0x80000000:0)|(se<<23)|frac;
        }
        bits[0]=(e<<23)|0x600000;bits[1]=0;bits[2]=0x80000000;
        if(!check(bits))return 1;
    }
    const uint32_t special[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0x7f800001,0xffc00000};
    for(auto v:special){std::fill(bits.begin(),bits.end(),v);if(!check(bits))return 1;}
    int layouts=0;
    for(int rows:{1,7,8,9,31,32,33,64}) for(int cols:{32,96,896,4864,8192}) for(bool p32:{false,true}) {
        std::vector<uint8_t> codes(size_t(rows)*cols),scales(size_t(rows)*(cols/32));
        for(auto& v:codes)v=uint8_t(rng());for(auto& v:scales)v=uint8_t(rng());
        auto ref=g3rt::Layout(rows,cols,p32).pack(codes.data(),scales.data());
        auto got=g3rt::pack_fast(rows,cols,p32,codes.data(),scales.data());
        if(ref.size()!=got.size() || std::memcmp(ref.data(),got.data(),ref.size()*sizeof(g3rt::Word))) {
            std::fprintf(stderr,"pack mismatch rows=%d cols=%d p32=%d\n",rows,cols,p32);return 1;
        }
        ++layouts;
    }
    std::printf("PASS bitexact_layouts=%d\n",layouts);
    std::printf("PASS bitexact_blocks=%llu values=%llu\n",(unsigned long long)blocks,(unsigned long long)(blocks*32));
}
